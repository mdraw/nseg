import numpy
import zstandard


def np_savezstd(fp: str, level=5, *args, **kwargs):
    """
    Save numpy arrays to a file using zstd compression.
    Like np.savez_compressed(), but with zstd instead of gzip.
    """

    with open(fp, 'wb') as f:
        cctx = zstandard.ZstdCompressor(level=level)
        with cctx.stream_writer(f) as compressor:
            numpy.savez(compressor, *args, **kwargs)


def np_loadzstd(fp: str):
    """
    Load numpy arrays from a file using zstd decompression.
    Like np.load(), but with zstd instead of gzip.
    """

    with open(fp, 'rb') as f:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(f) as decompressor:
            return numpy.load(decompressor)
