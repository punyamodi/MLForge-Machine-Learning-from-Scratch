from .pca import PCA
from .lda import LinearDiscriminantAnalysis
from .kernel_pca import KernelPCA
from .tsne import TSNE
from .nmf import NMF
from .truncated_svd import TruncatedSVD
from .ica import IndependentComponentAnalysis
from .factor_analysis import FactorAnalysis

__all__ = [
    "PCA", "LinearDiscriminantAnalysis", "KernelPCA", "TSNE",
    "NMF", "TruncatedSVD", "IndependentComponentAnalysis", "FactorAnalysis",
]
