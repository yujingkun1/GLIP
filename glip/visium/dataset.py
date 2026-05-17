import os
import random

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from glip.utils import normalize_expression
from . import config as CFG


def _decode_barcode(value):
    current = value
    while isinstance(current, np.ndarray):
        if current.ndim == 0:
            current = current.item()
            break
        if current.size == 1:
            current = current.reshape(-1)[0]
            continue
        break

    if isinstance(current, bytes):
        return current.decode("utf-8")

    return str(current)


def _close_backed_adata(adata):
    try:
        if getattr(adata, "isbacked", False) and getattr(adata, "file", None) is not None:
            adata.file.close()
    except Exception:
        pass


def _parse_sample_ids(sample_ids):
    if sample_ids is None:
        return None
    if isinstance(sample_ids, str):
        sample_ids = [sample_id.strip() for sample_id in sample_ids.split(",")]
    parsed = [sample_id for sample_id in sample_ids if sample_id]
    return parsed or None


def discover_hest_sample_ids(hest_data_dir):
    st_dir = os.path.join(hest_data_dir, "st")
    patch_dir = os.path.join(hest_data_dir, "patches")

    if not os.path.isdir(st_dir):
        raise FileNotFoundError(f"HEST ST directory not found: {st_dir}")
    if not os.path.isdir(patch_dir):
        raise FileNotFoundError(f"HEST patch directory not found: {patch_dir}")

    sample_ids = []
    for filename in sorted(os.listdir(st_dir)):
        if not filename.endswith(".h5ad"):
            continue
        sample_id = os.path.splitext(filename)[0]
        if os.path.exists(os.path.join(patch_dir, f"{sample_id}.h5")):
            sample_ids.append(sample_id)

    if not sample_ids:
        raise RuntimeError(
            f"No HEST samples with both st/*.h5ad and patches/*.h5 were found under: {hest_data_dir}"
        )

    return sample_ids


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_path=None,
        spatial_pos_path=None,
        barcode_path=None,
        reduced_mtx_path=None,
        hest_data_dir=None,
        sample_ids=None,
        gene_file=None,
        max_spots_per_sample=None,
        expression_normalization="log1p",
        cpm_scale=1_000_000.0,
        is_train=False,
        model_name=None,
    ):
        self.is_train = is_train
        self.mode = "legacy" if hest_data_dir is None else "hest"
        self.expression_normalization = str(expression_normalization).strip().lower()
        self.cpm_scale = float(cpm_scale)
        self.num_features = 0
        self.num_genes = 0
        self.selected_genes = []
        self._adata_cache = {}
        self._patch_cache = {}
        self.model_name = model_name  # Store model name for normalization

        if self.mode == "legacy":
            self._init_legacy_dataset(
                image_path=image_path,
                spatial_pos_path=spatial_pos_path,
                barcode_path=barcode_path,
                reduced_mtx_path=reduced_mtx_path,
            )
        else:
            self._init_hest_dataset(
                hest_data_dir=hest_data_dir,
                sample_ids=sample_ids,
                gene_file=gene_file,
                max_spots_per_sample=max_spots_per_sample,
            )

    def _init_legacy_dataset(self, image_path, spatial_pos_path, barcode_path, reduced_mtx_path):
        if not all([image_path, spatial_pos_path, barcode_path, reduced_mtx_path]):
            raise ValueError(
                "Legacy CLIPDataset requires image_path, spatial_pos_path, barcode_path, and reduced_mtx_path."
            )

        import cv2

        self.image_path = os.path.expanduser(image_path)
        self.spatial_pos_path = os.path.expanduser(spatial_pos_path)
        self.barcode_path = os.path.expanduser(barcode_path)
        self.reduced_mtx_path = os.path.expanduser(reduced_mtx_path)

        whole_image = cv2.imread(self.image_path)
        if whole_image is None:
            raise FileNotFoundError(f"Failed to read image file: {self.image_path}")

        self.whole_image = cv2.cvtColor(whole_image, cv2.COLOR_BGR2RGB)
        self.spatial_pos_csv = pd.read_csv(self.spatial_pos_path, sep=",", header=None)
        self.barcode_tsv = pd.read_csv(self.barcode_path, sep="\t", header=None)
        self.reduced_matrix = np.load(self.reduced_mtx_path).T.astype(np.float32)
        if self.expression_normalization != "log1p":
            self.reduced_matrix = np.stack(
                [
                    normalize_expression(row, self.expression_normalization, self.cpm_scale)
                    for row in self.reduced_matrix
                ],
                axis=0,
            ).astype(np.float32, copy=False)
        self.num_features = self.reduced_matrix.shape[1]
        self.num_genes = self.num_features

        print("Finished loading legacy BLEEP dataset files")
        print(f"Legacy dataset spots: {len(self.barcode_tsv)}")
        print(f"Legacy feature dimension: {self.num_features}")

    def _init_hest_dataset(self, hest_data_dir, sample_ids, gene_file, max_spots_per_sample):
        self.hest_data_dir = os.path.abspath(os.path.expanduser(hest_data_dir))
        self.sample_ids = _parse_sample_ids(sample_ids) or discover_hest_sample_ids(self.hest_data_dir)
        self.gene_file = os.path.expanduser(gene_file) if gene_file else None
        self.max_spots_per_sample = None if max_spots_per_sample in (None, 0) else int(max_spots_per_sample)
        self.sample_info = {}
        self.entries = []

        print("=== Initializing HEST-backed BLEEP dataset ===")
        print(f"HEST root: {self.hest_data_dir}")
        print(f"Samples requested: {len(self.sample_ids)}")

        self.selected_genes = self._resolve_selected_genes()
        self.num_features = len(self.selected_genes)
        self.num_genes = self.num_features

        if self.num_features == 0:
            raise RuntimeError("No genes were resolved for the HEST dataset.")

        self._build_hest_index()

        if not self.entries:
            raise RuntimeError("No aligned HEST spots were found after barcode matching.")

        print(f"HEST aligned spots: {len(self.entries)}")
        print(f"HEST gene dimension: {self.num_features}")

    def _resolve_selected_genes(self):
        if self.gene_file:
            if not os.path.exists(self.gene_file):
                raise FileNotFoundError(f"Gene file not found: {self.gene_file}")

            selected_genes = []
            seen = set()
            with open(self.gene_file, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    gene = raw_line.strip()
                    if not gene:
                        continue
                    if gene.startswith(("Efficiently", "Total", "Detection", "Samples")):
                        continue
                    if gene in seen:
                        continue
                    selected_genes.append(gene)
                    seen.add(gene)

            print(f"Using gene list from file: {self.gene_file}")
            print(f"Selected genes from file: {len(selected_genes)}")
            return selected_genes

        shared_gene_set = None
        gene_order_reference = None

        for sample_id in self.sample_ids:
            st_path = os.path.join(self.hest_data_dir, "st", f"{sample_id}.h5ad")
            if not os.path.exists(st_path):
                raise FileNotFoundError(f"HEST sample not found: {st_path}")

            adata = ad.read_h5ad(st_path, backed="r")
            sample_genes = [str(gene) for gene in adata.var_names]
            _close_backed_adata(adata)

            if gene_order_reference is None:
                gene_order_reference = sample_genes
                shared_gene_set = set(sample_genes)
            else:
                shared_gene_set &= set(sample_genes)

        selected_genes = [gene for gene in gene_order_reference if gene in shared_gene_set]
        print("Using the shared gene intersection across selected HEST samples")
        print(f"Shared genes: {len(selected_genes)}")
        return selected_genes

    def _build_hest_index(self):
        total_aligned = 0

        for sample_id in self.sample_ids:
            st_path = os.path.join(self.hest_data_dir, "st", f"{sample_id}.h5ad")
            patch_path = os.path.join(self.hest_data_dir, "patches", f"{sample_id}.h5")

            if not os.path.exists(st_path):
                print(f"[HEST] Skip sample without AnnData: {sample_id}")
                continue
            if not os.path.exists(patch_path):
                print(f"[HEST] Skip sample without patches: {sample_id}")
                continue

            adata = ad.read_h5ad(st_path, backed="r")
            obs_names = pd.Index([str(obs_name) for obs_name in adata.obs_names])
            var_names = pd.Index([str(gene) for gene in adata.var_names])

            gene_indexer = var_names.get_indexer(self.selected_genes)
            present_mask = gene_indexer >= 0
            if not present_mask.any():
                _close_backed_adata(adata)
                print(f"[HEST] Skip sample with zero selected genes present: {sample_id}")
                continue

            with h5py.File(patch_path, "r") as patch_file:
                patch_barcodes = [_decode_barcode(value) for value in patch_file["barcode"][:]]
                patch_coords = np.asarray(patch_file["coords"])
                patch_count = int(patch_file["img"].shape[0])

            aligned_size = min(len(patch_barcodes), len(patch_coords), patch_count)
            patch_barcodes = patch_barcodes[:aligned_size]
            patch_coords = patch_coords[:aligned_size]

            obs_indexer = obs_names.get_indexer(patch_barcodes)
            valid_patch_indices = np.flatnonzero(obs_indexer >= 0)
            if self.max_spots_per_sample is not None:
                valid_patch_indices = valid_patch_indices[: self.max_spots_per_sample]

            if len(valid_patch_indices) == 0:
                _close_backed_adata(adata)
                print(f"[HEST] Skip sample with zero barcode matches: {sample_id}")
                continue

            self.sample_info[sample_id] = {
                "st_path": st_path,
                "patch_path": patch_path,
                "gene_indexer": gene_indexer,
                "present_mask": present_mask,
            }

            for patch_idx in valid_patch_indices:
                coords = patch_coords[patch_idx]
                self.entries.append(
                    {
                        "sample_id": sample_id,
                        "patch_idx": int(patch_idx),
                        "obs_idx": int(obs_indexer[patch_idx]),
                        "barcode": patch_barcodes[patch_idx],
                        "spatial_coords": [int(coords[0]), int(coords[1])],
                    }
                )

            total_aligned += len(valid_patch_indices)
            _close_backed_adata(adata)
            print(
                f"[HEST] {sample_id}: matched {len(valid_patch_indices)} spots, "
                f"genes present {int(present_mask.sum())}/{len(self.selected_genes)}"
            )

        print(f"[HEST] Total aligned spots across samples: {total_aligned}")

    def _get_hest_adata(self, sample_id):
        adata = self._adata_cache.get(sample_id)
        if adata is None:
            adata = ad.read_h5ad(self.sample_info[sample_id]["st_path"], backed="r")
            self._adata_cache[sample_id] = adata
        return adata

    def _get_hest_patch_file(self, sample_id):
        patch_file = self._patch_cache.get(sample_id)
        if patch_file is None:
            patch_file = h5py.File(self.sample_info[sample_id]["patch_path"], "r")
            self._patch_cache[sample_id] = patch_file
        return patch_file

    def transform(self, image, is_train=None):
        use_train_aug = self.is_train if is_train is None else is_train
        image = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")

        if use_train_aug:
            if random.random() > 0.5:
                image = TF.hflip(image)
            if random.random() > 0.5:
                image = TF.vflip(image)
            image = TF.rotate(image, random.choice([180, 90, 0, -90]))

        image = TF.to_tensor(image)

        # Select normalization parameters based on model
        if self.model_name and 'h0mini' in str(self.model_name).lower():
            # H0-mini normalization
            mean = CFG.H0MINI_MEAN
            std = CFG.H0MINI_STD
        else:
            # ImageNet normalization (default for other models)
            mean = CFG.IMAGENET_MEAN
            std = CFG.IMAGENET_STD

        image = TF.normalize(image, mean=mean, std=std)
        return image

    def get_item(self, idx, is_train=None):
        if self.mode == "legacy":
            barcode = self.barcode_tsv.values[idx, 0]
            v1 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode, 4].values[0]
            v2 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode, 5].values[0]

            image = self.whole_image[(v1 - 112) : (v1 + 112), (v2 - 112) : (v2 + 112)]
            image = self.transform(image, is_train=is_train)

            return {
                "image": image.float(),
                "reduced_expression": torch.tensor(self.reduced_matrix[idx, :]).float(),
                "barcode": barcode,
                "spatial_coords": [v1, v2],
            }

        entry = self.entries[idx]
        sample_id = entry["sample_id"]
        sample_info = self.sample_info[sample_id]

        patch_file = self._get_hest_patch_file(sample_id)
        image = patch_file["img"][entry["patch_idx"]]
        image = self.transform(image, is_train=is_train)

        adata = self._get_hest_adata(sample_id)
        gene_expression = np.zeros(self.num_features, dtype=np.float32)

        present_mask = sample_info["present_mask"]
        if present_mask.any():
            gene_positions = np.flatnonzero(present_mask)
            gene_indices = sample_info["gene_indexer"][present_mask]
            sorted_order = np.argsort(gene_indices)
            sorted_gene_indices = gene_indices[sorted_order]
            sorted_gene_positions = gene_positions[sorted_order]

            expression_row = adata.X[entry["obs_idx"], sorted_gene_indices]
            if hasattr(expression_row, "toarray"):
                expression_row = expression_row.toarray()
            expression_row = np.asarray(expression_row).reshape(-1).astype(np.float32)
            gene_expression[sorted_gene_positions] = expression_row

        gene_expression = normalize_expression(gene_expression, self.expression_normalization, self.cpm_scale)

        return {
            "image": image.float(),
            "reduced_expression": torch.from_numpy(gene_expression),
            "barcode": entry["barcode"],
            "spatial_coords": entry["spatial_coords"],
            "sample_id": sample_id,
        }

    def __getitem__(self, idx):
        return self.get_item(idx, is_train=self.is_train)

    def __len__(self):
        if self.mode == "legacy":
            return len(self.barcode_tsv)
        return len(self.entries)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_adata_cache"] = {}
        state["_patch_cache"] = {}
        return state

    def __del__(self):
        for adata in getattr(self, "_adata_cache", {}).values():
            _close_backed_adata(adata)
        for patch_file in getattr(self, "_patch_cache", {}).values():
            try:
                patch_file.close()
            except Exception:
                pass


class CLIPSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, is_train=False):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.is_train = is_train
        self.num_features = dataset.num_features
        self.num_genes = dataset.num_genes
        self.selected_genes = dataset.selected_genes

    def __getitem__(self, idx):
        return self.dataset.get_item(self.indices[idx], is_train=self.is_train)

    def __len__(self):
        return len(self.indices)
