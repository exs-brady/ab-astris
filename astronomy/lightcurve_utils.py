"""
Lightcurve utility functions for handling multi-sector TESS data.

Addresses lightkurve .stitch() issues with inconsistent column types
across different TESS sectors.
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def safe_stitch(lc_collection):
    """
    Safely stitch a LightCurveCollection, handling inconsistent column types.

    Different TESS sectors can have different data types for the 'quality' column:
    - Sectors 1-50: typically int32
    - Sectors 51+: sometimes int16
    - Some sectors: str32 (!)

    This function normalizes all columns to consistent types before stitching.

    Args:
        lc_collection: lightkurve LightCurveCollection from download_all()

    Returns:
        Stitched LightCurve object
    """
    if len(lc_collection) == 0:
        raise ValueError("Empty lightcurve collection")

    if len(lc_collection) == 1:
        return lc_collection[0]

    # Normalize each lightcurve's columns before stitching
    normalized_lcs = []
    for i, lc in enumerate(lc_collection):
        try:
            # Get underlying table data
            if hasattr(lc, 'meta') and hasattr(lc, 'to_table'):
                table = lc.to_table()

                # Normalize 'quality' column to int32
                if 'quality' in table.colnames:
                    try:
                        # Convert to int32, handling any type
                        quality_data = table['quality']
                        if hasattr(quality_data, 'astype'):
                            table['quality'] = quality_data.astype(np.int32)
                        else:
                            # Handle string or other types
                            table['quality'] = np.array(quality_data, dtype=np.int32)
                    except (ValueError, TypeError) as e:
                        # If conversion fails, just use zeros (quality mask will handle it)
                        logger.warning(f"Could not convert quality column in sector {i}: {e}")
                        table['quality'] = np.zeros(len(table), dtype=np.int32)

                # Normalize 'cadenceno' column if present
                if 'cadenceno' in table.colnames:
                    try:
                        table['cadenceno'] = np.array(table['cadenceno'], dtype=np.int64)
                    except (ValueError, TypeError):
                        pass

                # Convert back to lightcurve
                from lightkurve import TessLightCurve
                normalized_lc = TessLightCurve(data=table, meta=lc.meta)
                normalized_lcs.append(normalized_lc)
            else:
                # Fallback: use original
                normalized_lcs.append(lc)

        except Exception as e:
            logger.warning(f"Failed to normalize lightcurve {i}: {e}, using original")
            normalized_lcs.append(lc)

    # Create new collection and stitch
    from lightkurve import LightCurveCollection
    collection = LightCurveCollection(normalized_lcs)

    return collection.stitch()


def download_and_stitch(search_result, quality_bitmask='default', max_sectors: Optional[int] = None):
    """
    Download lightcurves and safely stitch them.

    This is a drop-in replacement for:
        search_result.download_all().stitch()

    Args:
        search_result: lightkurve SearchResult from search_lightcurve()
        quality_bitmask: Quality bitmask for download (default: 'default')
        max_sectors: Maximum number of sectors to download (None = all).
                     For memory-constrained environments, use 4-6 sectors.
                     Downloads most recent sectors first (best data quality).

    Returns:
        Stitched LightCurve object

    Raises:
        ValueError: If no valid lightcurves could be downloaded
    """
    from pathlib import Path
    import shutil

    # Limit sectors if specified (take most recent for best data quality)
    if max_sectors is not None and len(search_result) > max_sectors:
        logger.info(f"Limiting download to {max_sectors} of {len(search_result)} available sectors")
        search_result = search_result[-max_sectors:]  # Take most recent sectors

    try:
        lc_collection = search_result.download_all(quality_bitmask=quality_bitmask)
        return safe_stitch(lc_collection)
    except Exception as e:
        error_msg = str(e).lower()
        # Check if this is a corrupted file error
        if 'corrupt' in error_msg or 'not recognized as a supported data product' in error_msg:
            logger.warning(f"Corrupted file detected, attempting to clear HLSP cache and retry: {e}")

            # Clear HLSP cache directory (where corrupted files typically live)
            cache_path = Path.home() / ".lightkurve" / "cache" / "mastDownload" / "HLSP"
            if cache_path.exists():
                try:
                    size_before = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
                    shutil.rmtree(cache_path)
                    logger.info(f"Cleared HLSP cache ({size_before / (1024*1024):.1f} MB)")
                except Exception as clear_err:
                    logger.warning(f"Failed to clear HLSP cache: {clear_err}")

            # Retry download
            logger.info("Retrying download after cache clear...")
            lc_collection = search_result.download_all(quality_bitmask=quality_bitmask)
            return safe_stitch(lc_collection)
        else:
            # Re-raise non-corruption errors
            raise
