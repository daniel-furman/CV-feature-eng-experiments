"""
Script to extract 128 x 128 pixel patches from chesapeake bay dataset
"""

import shutil 
import glob
from typing import Dict, List
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib.image
import os
import numpy as np
import pandas as pd
import copy
from torchgeo.datamodules import ChesapeakeCVPRDataModule
import torch
from torch.utils.data import DataLoader


# To use this script, you'll first need to download the 
# 150 GB of raw CVPR data, find it here:
# https://lila.science/datasets/chesapeakelandcover

# select a state to sample (see link for options)
state = "wv"

fpath_existing = glob.glob(f'data/patches_dataset/{state}_patches/**/**/*.npy')

ids = []
for f in fpath_existing:
    id_itr = f.split('_')[2].split('/')[-1]
    ids.append(int(id_itr))

itr_save = np.max(np.array(ids)) + 1
        
# Source path 
src = 'data/training_set_empty_template'
    
# Destination path 
dest = f'data/patches_dataset/{state}_patches'
    
# Copy the content of 
# source to destination 
# shutil.copytree(src, dest) 

d_mod = ChesapeakeCVPRDataModule(root_dir='/Volumes/Extreme Pro/CV_final/data/cvpr_chesapeake_landcover/',
                    train_splits=[f"{state}-train"],
                    val_splits=[f"{state}-val"],
                    test_splits=[f"{state}-test"],
                    patches_per_tile=200,
                    patch_size=128,
                    batch_size=64,
                    num_workers=0, 
                    class_set=7,
                    seed=1337,)


d_mod.prepare_data()
d_mod.setup(stage=None)
val = d_mod.val_dataloader()
train = d_mod.train_dataloader()
len(train)
len(val)
label_dict = {
    7: 'NaN',
    0: 'Water',#: All areas of open water including ponds, rivers, and lakes
    1: 'Tree Canopy and Shrubs',#: All woody vegetation including trees and shrubs
    2: 'Low Vegetation',#: Plant material less than 2 meters in height including lawns
    3: 'Barren',#: Areas devoid of vegetation consisting of natural earthen material
    4: 'Impervious Surfaces',#: Human-constructed surfaces less than 2 meters in height
    5: 'Impervious Roads',#: Impervious surfaces that are used for transportation
    6: 'Aberdeen Proving Ground',#: U.S. Army facility with no labels


}

label_counts = {
    'NaN': 0,
    'Water': 0,#: All areas of open water including ponds, rivers, and lakes
    'Tree Canopy and Shrubs': 0,#: All woody vegetation including trees and shrubs
    'Low Vegetation': 0,#: Plant material less than 2 meters in height including lawns
    'Barren': 0,#: Areas devoid of vegetation consisting of natural earthen material
    'Impervious Surfaces': 0,#: Human-constructed surfaces less than 2 meters in height
    'Impervious Roads': 0,#: Impervious surfaces that are used for transportation
    'Aberdeen Proving Ground': 0,#: U.S. Army facility with no labels
}
# extract nir, combine it with extracted rgb features 

n_batches = len(val)
for i in range(0,n_batches):
    batch = next(iter(val))
    batch['image'].shape
    batch['mask'].shape
    
    for itr, image in enumerate(batch['image']):


        mask = batch['mask'][itr]

        mask = mask.numpy()

        areas = []
        for i in range(1,7):

            areas.append(len(mask[mask==i]) / len(mask.flatten()))

        if max(areas) >= 0.85:

            areas = np.array(areas)

            rgb = image[0:3, :, :]
            rgb = rgb.numpy()
            if itr_save == 1:
                print(rgb.shape)
            rgb = np.swapaxes(rgb, 0, 2)
            rgb = np.swapaxes(rgb, 0, 1)
            label_counts[label_dict[np.argmax(areas)]]+=1
            label_name = label_dict[np.argmax(areas)]
            nir = image[3, :, :]
            if label_dict[np.argmax(areas)] not in ['Tree Canopy and Shrubs', 'Low Vegetation']:

                np.save(f'data/patches_dataset/{state}_patches/{label_name}/rgb/{itr_save}_rgb', rgb)
                np.save(f'data/patches_dataset/{state}_patches/{label_name}/nir/{itr_save}_nir', nir)
                matplotlib.image.imsave(f'data/patches_dataset/{state}_patches/{label_name}/images/{itr_save}_image.png', rgb)

                itr_save+=1

            else:

                if np.random.randint(0, 5) == 0:

                    np.save(f'data/patches_dataset/{state}_patches/{label_name}/rgb/{itr_save}_rgb', rgb)
                    np.save(f'data/patches_dataset/{state}_patches/{label_name}/nir/{itr_save}_nir', nir)
                    matplotlib.image.imsave(f'data/patches_dataset/{state}_patches/{label_name}/images/{itr_save}_image.png', rgb)

                    itr_save+=1

print(label_counts)




# Re-define a couple torchgeo scripts so they don't err out:

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Chesapeake Bay High-Resolution Land Cover Project datasets."""

import abc
import os
import sys
from typing import Any, Callable, Dict, Optional, Sequence

import fiona
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import rasterio.mask
import shapely.geometry
import shapely.ops
import torch
from matplotlib.colors import ListedColormap
from rasterio.crs import CRS

from torchgeo.datasets.geo import GeoDataset, RasterDataset
from torchgeo.datasets.utils import BoundingBox, download_url, extract_archive


class Chesapeake(RasterDataset, abc.ABC):
    """Abstract base class for all Chesapeake datasets.

    `Chesapeake Bay High-Resolution Land Cover Project
    <https://www.chesapeakeconservancy.org/conservation-innovation-center/high-resolution-data/land-cover-data-project/>`_
    dataset.

    This dataset was collected by the Chesapeake Conservancy's Conservation Innovation
    Center (CIC) in partnership with the University of Vermont and WorldView Solutions,
    Inc. It consists of one-meter resolution land cover information for the Chesapeake
    Bay watershed (~100,000 square miles of land).

    For more information, see:

    * `User Guide
      <https://chesapeakeconservancy.org/wp-content/uploads/2017/01/LandCover101Guide.pdf>`_
    * `Class Descriptions
      <https://chesapeakeconservancy.org/wp-content/uploads/2020/03/LC_Class_Descriptions.pdf>`_
    * `Accuracy Assessment
      <https://chesapeakeconservancy.org/wp-content/uploads/2017/01/Chesapeake_Conservancy_Accuracy_Assessment_Methodology.pdf>`_
    """

    is_image = False

    # subclasses use the 13 class cmap by default
    cmap = {
        0: (0, 0, 0, 0),
        1: (0, 197, 255, 255),
        2: (0, 168, 132, 255),
        3: (38, 115, 0, 255),
        4: (76, 230, 0, 255),
        5: (163, 255, 115, 255),
        6: (255, 170, 0, 255),
        7: (255, 0, 0, 255),
        8: (156, 156, 156, 255),
        9: (0, 0, 0, 255),
        10: (115, 115, 0, 255),
        11: (230, 230, 0, 255),
        12: (255, 255, 115, 255),
        13: (197, 0, 255, 255),
    }

    @property
    @abc.abstractmethod
    def base_folder(self) -> str:
        """Parent directory of dataset in URL."""

    @property
    @abc.abstractmethod
    def filename(self) -> str:
        """Filename to find/store dataset in."""

    @property
    @abc.abstractmethod
    def zipfile(self) -> str:
        """Name of zipfile in download URL."""

    @property
    @abc.abstractmethod
    def md5(self) -> str:
        """MD5 checksum to verify integrity of dataset."""

    @property
    def url(self) -> str:
        """URL to download dataset from."""
        url = "https://cicwebresources.blob.core.windows.net/chesapeakebaylandcover"
        url += f"/{self.base_folder}/{self.zipfile}"
        return url

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        self.root = root
        self.download = download
        self.checksum = checksum

        self._verify()

        colors = []
        for i in range(len(self.cmap)):
            colors.append(
                (
                    self.cmap[i][0] / 255.0,
                    self.cmap[i][1] / 255.0,
                    self.cmap[i][2] / 255.0,
                )
            )
        self._cmap = ListedColormap(colors)

        super().__init__(root, crs, res, transforms, cache)


    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted file already exists
        if os.path.exists(os.path.join(self.root, self.filename)):
            return

        # Check if the zip file has already been downloaded
        if os.path.exists(os.path.join(self.root, self.zipfile)):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        download_url(self.url, self.root, filename=self.zipfile, md5=self.md5)

    def _extract(self) -> None:
        """Extract the dataset."""
        extract_archive(os.path.join(self.root, self.zipfile))

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionchanged:: 0.3
           Method now takes a sample dict, not a Tensor. Additionally, possible to
           show subplot titles and/or use a custom suptitle.
        """
        mask = sample["mask"].squeeze(0)
        ncols = 1

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze(0)
            ncols = 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols, 4))

        if showing_predictions:
            axs[0].imshow(
                mask,
                vmin=0,
                vmax=self._cmap.N - 1,
                cmap=self._cmap,
                interpolation="none",
            )
            axs[0].axis("off")
            axs[1].imshow(
                pred,
                vmin=0,
                vmax=self._cmap.N - 1,
                cmap=self._cmap,
                interpolation="none",
            )
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title("Mask")
                axs[1].set_title("Prediction")

        else:
            axs.imshow(
                mask,
                vmin=0,
                vmax=self._cmap.N - 1,
                cmap=self._cmap,
                interpolation="none",
            )
            axs.axis("off")
            if show_titles:
                axs.set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig



class Chesapeake7(Chesapeake):
    """Complete 7-class dataset.

    This version of the dataset is composed of 7 classes:

    0. No Data: Background values
    1. Water: All areas of open water including ponds, rivers, and lakes
    2. Tree Canopy and Shrubs: All woody vegetation including trees and shrubs
    3. Low Vegetation: Plant material less than 2 meters in height including lawns
    4. Barren: Areas devoid of vegetation consisting of natural earthen material
    5. Impervious Surfaces: Human-constructed surfaces less than 2 meters in height
    6. Impervious Roads: Impervious surfaces that are used for transportation
    7. Aberdeen Proving Ground: U.S. Army facility with no labels
    """

    base_folder = "BAYWIDE"
    filename = "Baywide_7class_20132014.tif"
    filename_glob = filename
    zipfile = "Baywide_7Class_20132014.zip"
    md5 = "61a4e948fb2551840b6557ef195c2084"

    cmap = {
        0: (0, 0, 0, 0),
        1: (0, 197, 255, 255),
        2: (38, 115, 0, 255),
        3: (163, 255, 115, 255),
        4: (255, 170, 0, 255),
        5: (156, 156, 156, 255),
        6: (0, 0, 0, 255),
        7: (197, 0, 255, 255),
    }


class Chesapeake13(Chesapeake):
    """Complete 13-class dataset.

    This version of the dataset is composed of 13 classes:

    0. No Data: Background values
    1. Water: All areas of open water including ponds, rivers, and lakes
    2. Wetlands: Low vegetation areas located along marine or estuarine regions
    3. Tree Canopy: Deciduous and evergreen woody vegetation over 3-5 meters in height
    4. Shrubland: Heterogeneous woody vegetation including shrubs and young trees
    5. Low Vegetation: Plant material less than 2 meters in height including lawns
    6. Barren: Areas devoid of vegetation consisting of natural earthen material
    7. Structures: Human-constructed objects made of impervious materials
    8. Impervious Surfaces: Human-constructed surfaces less than 2 meters in height
    9. Impervious Roads: Impervious surfaces that are used for transportation
    10. Tree Canopy over Structures: Tree cover overlapping impervious structures
    11. Tree Canopy over Impervious Surfaces: Tree cover overlapping impervious surfaces
    12. Tree Canopy over Impervious Roads: Tree cover overlapping impervious roads
    13. Aberdeen Proving Ground: U.S. Army facility with no labels
    """

    base_folder = "BAYWIDE"
    filename = "Baywide_13Class_20132014.tif"
    filename_glob = filename
    zipfile = "Baywide_13Class_20132014.zip"
    md5 = "7e51118923c91e80e6e268156d25a4b9"


class ChesapeakeDC(Chesapeake):
    """This subset of the dataset contains data only for Washington, D.C."""

    base_folder = "DC"
    filename = os.path.join("DC_11001", "DC_11001.img")
    filename_glob = filename
    zipfile = "DC_11001.zip"
    md5 = "ed06ba7570d2955e8857d7d846c53b06"


class ChesapeakeDE(Chesapeake):
    """This subset of the dataset contains data only for Delaware."""

    base_folder = "DE"
    filename = "DE_STATEWIDE.tif"
    filename_glob = filename
    zipfile = "_DE_STATEWIDE.zip"
    md5 = "5e12eff3b6950c01092c7e480b38e544"


class ChesapeakeMD(Chesapeake):
    """This subset of the dataset contains data only for Maryland.

    .. note::

       This dataset requires the following additional library to be installed:

       * `zipfile-deflate64 <https://pypi.org/project/zipfile-deflate64/>`_ to extract
         the proprietary deflate64 compressed zip file.
    """

    base_folder = "MD"
    filename = "MD_STATEWIDE.tif"
    filename_glob = filename
    zipfile = "_MD_STATEWIDE.zip"
    md5 = "40c7cd697a887f2ffdb601b5c114e567"


class ChesapeakeNY(Chesapeake):
    """This subset of the dataset contains data only for New York.

    .. note::

       This dataset requires the following additional library to be installed:

       * `zipfile-deflate64 <https://pypi.org/project/zipfile-deflate64/>`_ to extract
         the proprietary deflate64 compressed zip file.
    """

    base_folder = "NY"
    filename = "NY_STATEWIDE.tif"
    filename_glob = filename
    zipfile = "_NY_STATEWIDE.zip"
    md5 = "1100078c526616454ef2e508affda915"


class ChesapeakePA(Chesapeake):
    """This subset of the dataset contains data only for Pennsylvania."""

    base_folder = "PA"
    filename = "PA_STATEWIDE.tif"
    filename_glob = filename
    zipfile = "_PA_STATEWIDE.zip"
    md5 = "20a2a857c527a4dbadd6beed8b47e5ab"


class ChesapeakeVA(Chesapeake):
    """This subset of the dataset contains data only for Virginia.

    .. note::

       This dataset requires the following additional library to be installed:

       * `zipfile-deflate64 <https://pypi.org/project/zipfile-deflate64/>`_ to extract
         the proprietary deflate64 compressed zip file.
    """

    base_folder = "VA"
    filename = "CIC2014_VA_STATEWIDE.tif"
    filename_glob = filename
    zipfile = "_VA_STATEWIDE.zip"
    md5 = "6f2c97deaf73bb3e1ea9b21bd7a3fc8e"


class ChesapeakeWV(Chesapeake):
    """This subset of the dataset contains data only for West Virginia."""

    base_folder = "WV"
    filename = "WV_STATEWIDE.tif"
    filename_glob = filename
    zipfile = "_WV_STATEWIDE.zip"
    md5 = "350621ea293651fbc557a1c3e3c64cc3"


class ChesapeakeCVPR(GeoDataset):
    """CVPR 2019 Chesapeake Land Cover dataset.

    The `CVPR 2019 Chesapeake Land Cover
    <https://lila.science/datasets/chesapeakelandcover>`_ dataset contains two layers of
    NAIP aerial imagery, Landsat 8 leaf-on and leaf-off imagery, Chesapeake Bay land
    cover labels, NLCD land cover labels, and Microsoft building footprint labels.

    This dataset was organized to accompany the 2019 CVPR paper, "Large Scale
    High-Resolution Land Cover Mapping with Multi-Resolution Data".

    The paper "Resolving label uncertainty with implicit generative models" added an
    additional layer of data to this dataset containing a prior over the Chesapeake Bay
    land cover classes generated from the NLCD land cover labels. For more information
    about this layer see `the dataset documentation
    <https://zenodo.org/record/5652512#.YcuAIZLMIQ8>`_.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/cvpr.2019.01301
    """

    subdatasets = ["base", "prior_extension"]
    urls = {
        "base": "https://lilablobssc.blob.core.windows.net/lcmcvpr2019/cvpr_chesapeake_landcover.zip",  # noqa: E501
        "prior_extension": "https://zenodo.org/record/5866525/files/cvpr_chesapeake_landcover_prior_extension.zip?download=1",  # noqa: E501
    }
    filenames = {
        "base": "cvpr_chesapeake_landcover.zip",
        "prior_extension": "cvpr_chesapeake_landcover_prior_extension.zip",
    }
    md5s = {
        "base": "1225ccbb9590e9396875f221e5031514",
        "prior_extension": "402f41d07823c8faf7ea6960d7c4e17a",
    }

    crs = CRS.from_epsg(3857)
    res = 1

    valid_layers = [
        "naip-new",
        "naip-old",
        "landsat-leaf-on",
        "landsat-leaf-off",
        "nlcd",
        "lc",
        "buildings",
        "prior_from_cooccurrences_101_31_no_osm_no_buildings",
    ]
    states = ["de", "md", "va", "wv", "pa", "ny"]
    splits = (
        [f"{state}-train" for state in states]
        + [f"{state}-val" for state in states]
        + [f"{state}-test" for state in states]
    )

    # these are used to check the integrity of the dataset
    files = [
        "de_1m_2013_extended-debuffered-test_tiles",
        "de_1m_2013_extended-debuffered-train_tiles",
        "de_1m_2013_extended-debuffered-val_tiles",
        "md_1m_2013_extended-debuffered-test_tiles",
        "md_1m_2013_extended-debuffered-train_tiles",
        "md_1m_2013_extended-debuffered-val_tiles",
        "ny_1m_2013_extended-debuffered-test_tiles",
        "ny_1m_2013_extended-debuffered-train_tiles",
        "ny_1m_2013_extended-debuffered-val_tiles",
        "pa_1m_2013_extended-debuffered-test_tiles",
        "pa_1m_2013_extended-debuffered-train_tiles",
        "pa_1m_2013_extended-debuffered-val_tiles",
        "va_1m_2014_extended-debuffered-test_tiles",
        "va_1m_2014_extended-debuffered-train_tiles",
        "va_1m_2014_extended-debuffered-val_tiles",
        "wv_1m_2014_extended-debuffered-test_tiles",
        "wv_1m_2014_extended-debuffered-train_tiles",
        "wv_1m_2014_extended-debuffered-val_tiles",
        "wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_buildings.tif",
        "wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_landsat-leaf-off.tif",  # noqa: E501
        "wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_landsat-leaf-on.tif",  # noqa: E501
        "wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_lc.tif",
        "wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_naip-new.tif",
        "wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_naip-old.tif",
        "wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_nlcd.tif",
        #"wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_prior_from_cooccurrences_101_31_no_osm_no_buildings.tif",  # noqa: E501
        "spatial_index.geojson",
    ]

    p_src_crs = pyproj.CRS("epsg:3857")
    p_transformers = {
        "epsg:26917": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26917"), always_xy=True
        ).transform,
        "epsg:26918": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26918"), always_xy=True
        ).transform,
    }

    def __init__(
        self,
        root: str = "data",
        splits: Sequence[str] = ["de-train"],
        layers: Sequence[str] = ["naip-new", "lc"],
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            splits: a list of strings in the format "{state}-{train,val,test}"
                indicating the subset of data to use, for example "ny-train"
            layers: a list containing a subset of "naip-new", "naip-old", "lc", "nlcd",
                "landsat-leaf-on", "landsat-leaf-off", "buildings", or
                "prior_from_cooccurrences_101_31_no_osm_no_buildings" indicating which
                layers to load
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
            AssertionError: if ``splits`` or ``layers`` are not valid
        """
        for split in splits:
            assert split in self.splits
        assert all([layer in self.valid_layers for layer in layers])
        self.root = root
        self.layers = layers
        self.cache = cache
        self.download = download
        self.checksum = checksum

        self._verify()

        super().__init__(transforms)

        # Add all tiles into the index in epsg:3857 based on the included geojson
        mint: float = 0
        maxt: float = sys.maxsize
        with fiona.open(os.path.join(root, "spatial_index.geojson"), "r") as f:
            for i, row in enumerate(f):
                if row["properties"]["split"] in splits:
                    box = shapely.geometry.shape(row["geometry"])
                    minx, miny, maxx, maxy = box.bounds
                    coords = (minx, maxx, miny, maxy, mint, maxt)

                    prior_fn = row["properties"]["lc"].replace(
                        "lc.tif",
                        "prior_from_cooccurrences_101_31_no_osm_no_buildings.tif",
                    )

                    self.index.insert(
                        i,
                        coords,
                        {
                            "naip-new": row["properties"]["naip-new"],
                            "naip-old": row["properties"]["naip-old"],
                            "landsat-leaf-on": row["properties"]["landsat-leaf-on"],
                            "landsat-leaf-off": row["properties"]["landsat-leaf-off"],
                            "lc": row["properties"]["lc"],
                            "nlcd": row["properties"]["nlcd"],
                            "buildings": row["properties"]["buildings"],
                            "prior_from_cooccurrences_101_31_no_osm_no_buildings": prior_fn,  # noqa: E501
                        },
                    )


    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = [hit.object for hit in hits]

        sample = {"image": [], "mask": [], "crs": self.crs, "bbox": query}

        if len(filepaths) == 0:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )
        elif len(filepaths) == 1:
            filenames = filepaths[0]
            query_geom_transformed = None  # is set by the first layer

            minx, maxx, miny, maxy, mint, maxt = query
            query_box = shapely.geometry.box(minx, miny, maxx, maxy)

            for layer in self.layers:

                fn = filenames[layer]

                with rasterio.open(os.path.join(self.root, fn)) as f:
                    dst_crs = f.crs.to_string().lower()

                    if query_geom_transformed is None:
                        query_box_transformed = shapely.ops.transform(
                            self.p_transformers[dst_crs], query_box
                        ).envelope
                        query_geom_transformed = shapely.geometry.mapping(
                            query_box_transformed
                        )

                    data, _ = rasterio.mask.mask(
                        f, [query_geom_transformed], crop=True, all_touched=True
                    )

                if layer in [
                    "naip-new",
                    "naip-old",
                    "landsat-leaf-on",
                    "landsat-leaf-off",
                ]:
                    sample["image"].append(data)
                elif layer in [
                    "lc",
                    "nlcd",
                    "buildings",
                    "prior_from_cooccurrences_101_31_no_osm_no_buildings",
                ]:
                    sample["mask"].append(data)
        else:
            raise IndexError(f"query: {query} spans multiple tiles which is not valid")

        sample["image"] = np.concatenate(sample["image"], axis=0)
        sample["mask"] = np.concatenate(sample["mask"], axis=0)

        sample["image"] = torch.from_numpy(sample["image"])
        sample["mask"] = torch.from_numpy(sample["mask"])

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        def exists(filename: str) -> bool:
            return os.path.exists(os.path.join(self.root, filename))

        if all(map(exists, self.files)):
            return

        # Check if the zip files have already been downloaded
        if all(
            [
                os.path.exists(os.path.join(self.root, self.filenames[subdataset]))
                for subdataset in self.subdatasets
            ]
        ):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        for subdataset in self.subdatasets:
            download_url(
                self.urls[subdataset],
                self.root,
                filename=self.filenames[subdataset],
                md5=self.md5s[subdataset],
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        for subdataset in self.subdatasets:
            extract_archive(os.path.join(self.root, self.filenames[subdataset]))
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Chesapeake Bay High-Resolution Land Cover Project datamodule."""

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from pytorch_lightning.core.datamodule import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from torchgeo.datasets import stack_samples
from torchgeo.samplers.batch import RandomBatchGeoSampler
from torchgeo.samplers.single import GridGeoSampler

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class ChesapeakeCVPRDataModule(LightningDataModule):
    """LightningDataModule implementation for the Chesapeake CVPR Land Cover dataset.

    Uses the random splits defined per state to partition tiles into train, val,
    and test sets.
    """

    def __init__(
        self,
        root_dir: str,
        train_splits: List[str],
        val_splits: List[str],
        test_splits: List[str],
        patches_per_tile: int = 200,
        patch_size: int = 256,
        batch_size: int = 64,
        num_workers: int = 0,
        class_set: int = 7,
        use_prior_labels: bool = False,
        prior_smoothing_constant: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for Chesapeake CVPR based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the ChesapeakeCVPR Dataset
                classes
            train_splits: The splits used to train the model, e.g. ["ny-train"]
            val_splits: The splits used to validate the model, e.g. ["ny-val"]
            test_splits: The splits used to test the model, e.g. ["ny-test"]
            patches_per_tile: The number of patches per tile to sample
            patch_size: The size of each patch in pixels (test patches will be 1.5 times
                this size)
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            class_set: The high-resolution land cover class set to use - 5 or 7
            use_prior_labels: Flag for using a prior over high-resolution classes
                instead of the high-resolution labels themselves
            prior_smoothing_constant: additive smoothing to add when using prior labels

        Raises:
            ValueError: if ``use_prior_labels`` is used with ``class_set==7``
        """
        super().__init__()
        for state in train_splits + val_splits + test_splits:
            assert state in ChesapeakeCVPR.splits
        assert class_set in [5, 7]
        if use_prior_labels and class_set != 5:
            raise ValueError(
                "The pre-generated prior labels are only valid for the 5"
                + " class set of labels"
            )

        self.root_dir = root_dir
        self.train_splits = train_splits
        self.val_splits = val_splits
        self.test_splits = test_splits
        self.patches_per_tile = patches_per_tile
        self.patch_size = patch_size
        # This is a rough estimate of how large of a patch we will need to sample in
        # EPSG:3857 in order to guarantee a large enough patch in the local CRS.
        self.original_patch_size = patch_size * 2
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_set = class_set
        self.use_prior_labels = use_prior_labels
        self.prior_smoothing_constant = prior_smoothing_constant

        if self.use_prior_labels:
            self.layers = [
                "naip-new",
                "prior_from_cooccurrences_101_31_no_osm_no_buildings",
            ]
        else:
            self.layers = ["naip-new", "lc"]


    def pad_to(
        self, size: int = 512, image_value: int = 0, mask_value: int = 0
    ) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Returns a function to perform a padding transform on a single sample.

        Args:
            size: output image size
            image_value: value to pad image with
            mask_value: value to pad mask with

        Returns:
            function to perform padding
        """

        def pad_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
            _, height, width = sample["image"].shape
            assert height <= size and width <= size

            height_pad = size - height
            width_pad = size - width

            # See https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            # for a description of the format of the padding tuple
            sample["image"] = F.pad(
                sample["image"],
                (0, width_pad, 0, height_pad),
                mode="constant",
                value=image_value,
            )
            sample["mask"] = F.pad(
                sample["mask"],
                (0, width_pad, 0, height_pad),
                mode="constant",
                value=mask_value,
            )
            return sample

        return pad_inner


    def center_crop(
        self, size: int = 512
    ) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Returns a function to perform a center crop transform on a single sample.

        Args:
            size: output image size

        Returns:
            function to perform center crop
        """

        def center_crop_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
            _, height, width = sample["image"].shape

            y1 = round((height - size) / 2)
            x1 = round((width - size) / 2)
            sample["image"] = sample["image"][:, y1 : y1 + size, x1 : x1 + size]
            sample["mask"] = sample["mask"][:, y1 : y1 + size, x1 : x1 + size]

            return sample

        return center_crop_inner


    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocesses a single sample.

        Args:
            sample: sample dictionary containing image and mask

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["image"] /= 255.0

        if "mask" in sample:
            sample["mask"] = sample["mask"].squeeze()
            if self.use_prior_labels:
                sample["mask"] = F.normalize(sample["mask"].float(), p=1, dim=0)
                sample["mask"] = F.normalize(
                    sample["mask"] + self.prior_smoothing_constant, p=1, dim=0
                )
            else:
                if self.class_set == 5:
                    sample["mask"][sample["mask"] == 5] = 4
                    sample["mask"][sample["mask"] == 6] = 4
                sample["mask"] = sample["mask"].long()

        return sample


    def remove_bbox(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Removes the bounding box property from a sample.

        Args:
            sample: dictionary with geographic metadata

        Returns
            sample without the bbox property
        """
        del sample["bbox"]
        return sample


    def nodata_check(
        self, size: int = 512
    ) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Returns a function to check for nodata or mis-sized input.

        Args:
            size: output image size

        Returns:
            function to check for nodata values
        """

        def nodata_check_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
            num_channels, height, width = sample["image"].shape

            if height < size or width < size:
                sample["image"] = torch.zeros((num_channels, size, size))
                sample["mask"] = torch.zeros((size, size))

            return sample

        return nodata_check_inner


    def prepare_data(self) -> None:
        """Confirms that the dataset is downloaded on the local node.

        This method is called once per node, while :func:`setup` is called once per GPU.
        """
        ChesapeakeCVPR(
            self.root_dir,
            splits=self.train_splits,
            layers=self.layers,
            transforms=None,
            download=False,
            checksum=False,
        )


    def setup(self, stage: Optional[str] = None) -> None:
        """Create the train/val/test splits based on the original Dataset objects.

        The splits should be done here vs. in :func:`__init__` per the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup.

        Args:
            stage: stage to set up
        """
        train_transforms = Compose(
            [
                self.center_crop(self.patch_size),
                self.nodata_check(self.patch_size),
                self.preprocess,
                self.remove_bbox,
            ]
        )
        val_transforms = Compose(
            [
                self.center_crop(self.patch_size),
                self.nodata_check(self.patch_size),
                self.preprocess,
                self.remove_bbox,
            ]
        )
        test_transforms = Compose(
            [
                self.pad_to(self.original_patch_size, image_value=0, mask_value=0),
                self.preprocess,
                self.remove_bbox,
            ]
        )

        self.train_dataset = ChesapeakeCVPR(
            self.root_dir,
            splits=self.train_splits,
            layers=self.layers,
            transforms=train_transforms,
            download=False,
            checksum=False,
        )
        self.val_dataset = ChesapeakeCVPR(
            self.root_dir,
            splits=self.val_splits,
            layers=self.layers,
            transforms=val_transforms,
            download=False,
            checksum=False,
        )
        self.test_dataset = ChesapeakeCVPR(
            self.root_dir,
            splits=self.test_splits,
            layers=self.layers,
            transforms=test_transforms,
            download=False,
            checksum=False,
        )


    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        sampler = RandomBatchGeoSampler(
            self.train_dataset,
            size=self.original_patch_size,
            batch_size=self.batch_size,
            length=self.patches_per_tile * len(self.train_dataset),
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )


    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        sampler = GridGeoSampler(
            self.val_dataset,
            size=self.original_patch_size,
            stride=self.original_patch_size,
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )


    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        sampler = GridGeoSampler(
            self.test_dataset,
            size=self.original_patch_size,
            stride=self.original_patch_size,
        )
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )
