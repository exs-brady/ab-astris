"""
Catalog Checker - VSX, SIMBAD, and Fetherolf cross-referencing
Determines if a TIC is a known variable star or a new discovery
"""

import requests
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta
import logging
from urllib.parse import quote
import pandas as pd
from pathlib import Path

try:
    from astroquery.simbad import Simbad
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ASTROQUERY_AVAILABLE = True
except ImportError:
    ASTROQUERY_AVAILABLE = False
    logging.warning("astroquery not available - SIMBAD queries will use fallback method")

from abastris.db import queries
from abastris.utils.config import settings

logger = logging.getLogger(__name__)


class CatalogChecker:
    """
    Check if a TIC is a known variable star in VSX or SIMBAD catalogs

    Uses caching to avoid repeated API calls (30-day cache)
    """

    def __init__(self, use_cache: bool = True):
        """
        Initialize catalog checker

        Args:
            use_cache: Whether to use Supabase cache (recommended)
        """
        self.use_cache = use_cache
        self.vsx_base_url = "https://www.aavso.org/vsx/index.php"

        # Initialize SIMBAD if available
        if ASTROQUERY_AVAILABLE:
            self.simbad = Simbad()
            # Add fields we want
            self.simbad.add_votable_fields('otype', 'otypes', 'ids')
        else:
            self.simbad = None

        # Load Fetherolf catalog (cached in memory)
        self.fetherolf_catalog = self._load_fetherolf_catalog()

    def check_tic(self, tic_number: int) -> Dict[str, Any]:
        """
        Complete catalog check for a TIC

        Args:
            tic_number: TESS Input Catalog identifier

        Returns:
            Dictionary with:
                - is_known_variable: bool
                - vsx_result: dict or None
                - simbad_result: dict or None
                - known_names: list of alternate names
                - variable_type: str or None
                - source: where it was found ('vsx', 'simbad', 'both', 'none')
        """
        logger.info(f"🔍 Checking catalogs for TIC {tic_number}")

        # Check cache first
        if self.use_cache:
            cached = self._get_cached_result(tic_number)
            if cached:
                logger.info(f"  ✅ Found in cache (age: {cached.get('cache_age', 'unknown')})")
                return cached

        # Query all catalogs
        vsx_result = self.query_vsx(tic_number)
        simbad_result = self.query_simbad(tic_number)
        fetherolf_result = self.check_fetherolf(tic_number)

        # Determine if known variable
        is_known_vsx = vsx_result is not None and vsx_result.get('found', False)
        is_known_simbad = simbad_result is not None and simbad_result.get('is_variable', False)
        is_known_fetherolf = fetherolf_result is not None and fetherolf_result.get('found', False)
        is_known_variable = is_known_vsx or is_known_simbad or is_known_fetherolf

        # Collect all known names
        known_names = []
        if vsx_result and 'name' in vsx_result:
            known_names.append(vsx_result['name'])
        if simbad_result and 'names' in simbad_result:
            known_names.extend(simbad_result['names'])
        if fetherolf_result:
            known_names.append(f"Fetherolf+2023 TIC {tic_number}")

        # Determine variable type
        variable_type = None
        if vsx_result and 'type' in vsx_result:
            variable_type = vsx_result['type']
        elif simbad_result and 'object_type' in simbad_result:
            variable_type = simbad_result['object_type']
        elif fetherolf_result and 'variable_type' in fetherolf_result:
            variable_type = fetherolf_result['variable_type']

        # Determine source
        sources = []
        if is_known_vsx:
            sources.append('vsx')
        if is_known_simbad:
            sources.append('simbad')
        if is_known_fetherolf:
            sources.append('fetherolf')

        if len(sources) == 0:
            source = 'none'
        elif len(sources) == 1:
            source = sources[0]
        else:
            source = '+'.join(sources)

        result = {
            'tic_number': tic_number,
            'is_known_variable': is_known_variable,
            'vsx_result': vsx_result,
            'simbad_result': simbad_result,
            'fetherolf_result': fetherolf_result,
            'known_names': list(set(known_names)),  # Remove duplicates
            'variable_type': variable_type,
            'source': source,
            'checked_at': datetime.now(timezone.utc).isoformat()
        }

        # Cache the result
        if self.use_cache and settings.is_configured:
            self._cache_result(result)

        # Log result
        if is_known_variable:
            logger.warning(f"  ⚠️  KNOWN VARIABLE: {known_names[0] if known_names else 'unnamed'}")
            logger.warning(f"     Type: {variable_type}, Source: {source}")
        else:
            logger.info(f"  ✅ NOT in catalogs - POTENTIAL NEW DISCOVERY!")

        return result

    def check_tic_with_coordinates(
        self,
        tic_number: int,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        search_radius: float = 5.0
    ) -> Dict[str, Any]:
        """
        Enhanced catalog check using coordinates as primary method

        This is more robust than name-only searches because:
        - Stars may be catalogued under different designations
        - Coordinates are universal across all catalogs
        - Reduces false negatives from naming mismatches

        Args:
            tic_number: TESS Input Catalog identifier
            ra: Right Ascension in decimal degrees (optional, will extract from TESS if not provided)
            dec: Declination in decimal degrees (optional, will extract from TESS if not provided)
            search_radius: Search radius in arcseconds (default 5.0)

        Returns:
            Dictionary with catalog results (same format as check_tic)
        """
        logger.info(f"🔍 Coordinate-based check for TIC {tic_number}")

        # Check cache first
        if self.use_cache:
            cached = self._get_cached_result(tic_number)
            if cached:
                logger.info(f"  ✅ Found in cache (age: {cached.get('cache_age', 'unknown')})")
                return cached

        # If coordinates not provided, try to get from TESS data
        if ra is None or dec is None:
            logger.debug(f"  Coordinates not provided, attempting to extract from TESS data...")
            coords = self._get_coordinates_from_tess(tic_number)
            if coords:
                ra, dec = coords
                logger.debug(f"  Extracted coordinates: RA={ra:.6f}°, Dec={dec:.6f}°")
            else:
                logger.warning(f"  Could not extract coordinates, falling back to name-only search")
                return self.check_tic(tic_number)  # Fallback to name-based search

        # Query both catalogs using coordinates
        vsx_result = self.query_vsx_by_coordinates(ra, dec, search_radius)
        simbad_result = self.query_simbad_by_coordinates(ra, dec, search_radius)

        # Determine if known variable
        is_known_vsx = vsx_result is not None and vsx_result.get('found', False)
        is_known_simbad = simbad_result is not None and simbad_result.get('is_variable', False)
        is_known_variable = is_known_vsx or is_known_simbad

        # Collect all known names
        known_names = []
        if vsx_result and 'name' in vsx_result:
            known_names.append(vsx_result['name'])
        if simbad_result and 'names' in simbad_result:
            known_names.extend(simbad_result['names'])

        # Determine variable type
        variable_type = None
        if vsx_result and 'type' in vsx_result:
            variable_type = vsx_result['type']
        elif simbad_result and 'object_type' in simbad_result:
            variable_type = simbad_result['object_type']

        # Determine source
        if is_known_vsx and is_known_simbad:
            source = 'both'
        elif is_known_vsx:
            source = 'vsx'
        elif is_known_simbad:
            source = 'simbad'
        else:
            source = 'none'

        result = {
            'tic_number': tic_number,
            'is_known_variable': is_known_variable,
            'vsx_result': vsx_result,
            'simbad_result': simbad_result,
            'known_names': list(set(known_names)),
            'variable_type': variable_type,
            'source': source,
            'search_method': 'coordinates',
            'search_coordinates': {'ra': ra, 'dec': dec},
            'search_radius_arcsec': search_radius,
            'checked_at': datetime.now(timezone.utc).isoformat()
        }

        # Cache the result
        if self.use_cache and settings.is_configured:
            self._cache_result(result)

        # Log result
        if is_known_variable:
            logger.warning(f"  ⚠️  KNOWN VARIABLE: {known_names[0] if known_names else 'unnamed'}")
            logger.warning(f"     Type: {variable_type}, Source: {source}")
            logger.warning(f"     Matched within {search_radius}\" of TIC coordinates")
        else:
            logger.info(f"  ✅ NOT in catalogs - POTENTIAL NEW DISCOVERY!")
            logger.info(f"     Searched {search_radius}\" radius around RA={ra:.6f}°, Dec={dec:.6f}°")

        return result

    def _get_coordinates_from_tess(self, tic_number: int) -> Optional[tuple]:
        """
        Extract coordinates from TESS light curve metadata

        Args:
            tic_number: TIC identifier

        Returns:
            Tuple of (ra, dec) in decimal degrees, or None if unavailable
        """
        try:
            import lightkurve as lk

            logger.debug(f"  Fetching TESS data to extract coordinates...")

            # Search for target
            search_result = lk.search_lightcurve(
                f"TIC {tic_number}",
                mission='TESS',
                author='SPOC'
            )

            if len(search_result) == 0:
                logger.debug(f"  No TESS data found for TIC {tic_number}")
                return None

            # Download first available light curve
            lc_collection = search_result[0].download()

            if lc_collection is None:
                return None

            # Get first light curve from collection
            lc = lc_collection[0] if hasattr(lc_collection, '__iter__') else lc_collection

            # Extract coordinates from metadata
            ra = lc.meta.get('RA_OBJ') or lc.meta.get('RA')
            dec = lc.meta.get('DEC_OBJ') or lc.meta.get('DEC')

            if ra is not None and dec is not None:
                return (float(ra), float(dec))

            logger.debug(f"  Coordinates not in TESS metadata for TIC {tic_number}")
            return None

        except Exception as e:
            logger.error(f"  Error extracting coordinates for TIC {tic_number}: {e}")
            return None

    def query_vsx_by_coordinates(
        self,
        ra: float,
        dec: float,
        radius: float = 5.0
    ) -> Optional[Dict[str, Any]]:
        """
        Query VSX using coordinate cone search

        Args:
            ra: Right Ascension in decimal degrees
            dec: Declination in decimal degrees
            radius: Search radius in arcseconds

        Returns:
            Dictionary with VSX data or None if not found
        """
        try:
            # Convert radius from arcseconds to arcminutes for VSX API
            radius_arcmin = radius / 60.0

            # VSX coordinate search endpoint
            # Format: coords=RA,DEC&format=csv&radius=arcmin
            url = (
                f"{self.vsx_base_url}?view=results.csv"
                f"&coords={ra:.6f},{dec:.6f}"
                f"&radius={radius_arcmin:.4f}"
            )

            logger.debug(f"  Querying VSX by coordinates: RA={ra:.6f}°, Dec={dec:.6f}°, radius={radius}\"")

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            content = response.text

            # Check if any results returned
            # VSX CSV format has header line, then data lines
            lines = content.strip().split('\n')

            if len(lines) <= 1:
                # Only header or empty
                logger.debug(f"  VSX: No variables found within {radius}\" of coordinates")
                return {'found': False}

            # Parse first result (closest match)
            # CSV format: Name,RA,Dec,Type,MaxMag,MinMag,Period,...
            data_line = lines[1].split(',')

            if len(data_line) < 3:
                logger.debug(f"  VSX: Invalid response format")
                return {'found': False}

            result = {
                'found': True,
                'name': data_line[0].strip(),
                'ra': float(data_line[1]) if data_line[1] else None,
                'dec': float(data_line[2]) if data_line[2] else None,
                'type': data_line[3].strip() if len(data_line) > 3 else None,
                'max_mag': data_line[4].strip() if len(data_line) > 4 else None,
                'min_mag': data_line[5].strip() if len(data_line) > 5 else None,
                'period': data_line[6].strip() if len(data_line) > 6 else None,
                'url': url,
                'num_matches': len(lines) - 1  # Total matches in search radius
            }

            logger.info(f"  VSX: Found {result['name']} (Type: {result['type']})")
            if result['num_matches'] > 1:
                logger.info(f"       ({result['num_matches']} total matches in search radius)")

            return result

        except requests.exceptions.Timeout:
            logger.error(f"  VSX: Timeout for coordinate search")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"  VSX: Error in coordinate search: {e}")
            return None
        except Exception as e:
            logger.error(f"  VSX: Unexpected error parsing response: {e}")
            return None

    def query_simbad_by_coordinates(
        self,
        ra: float,
        dec: float,
        radius: float = 5.0
    ) -> Optional[Dict[str, Any]]:
        """
        Query SIMBAD using coordinate cone search

        Args:
            ra: Right Ascension in decimal degrees
            dec: Declination in decimal degrees
            radius: Search radius in arcseconds

        Returns:
            Dictionary with SIMBAD data or None if not found
        """
        if not ASTROQUERY_AVAILABLE:
            logger.warning("  SIMBAD: astroquery not available, skipping coordinate search")
            return None

        try:
            # Create SkyCoord object
            coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

            logger.debug(f"  Querying SIMBAD by coordinates: RA={ra:.6f}°, Dec={dec:.6f}°, radius={radius}\"")

            # Query region
            result_table = self.simbad.query_region(
                coord,
                radius=radius*u.arcsec
            )

            if result_table is None or len(result_table) == 0:
                logger.debug(f"  SIMBAD: No objects found within {radius}\" of coordinates")
                return {'found': False}

            # Check all results for variable stars (not just first)
            for row in result_table:
                # Get object type - handle missing OTYPE field gracefully
                try:
                    if 'OTYPE' in row.colnames:
                        object_type = row['OTYPE'].decode('utf-8') if isinstance(row['OTYPE'], bytes) else str(row['OTYPE'])
                    else:
                        object_type = 'Unknown'
                except (KeyError, AttributeError):
                    object_type = 'Unknown'

                # Check if it's a variable star type
                variable_types = ['V*', 'EB', 'Pulsating', 'Eruptive', 'Cataclysmic', 'BY', 'RS']
                is_variable = any(vt in object_type for vt in variable_types)

                if is_variable:
                    # Found a variable star

                    # Get all identifiers/names
                    names = []
                    if 'IDS' in row.colnames:
                        ids_str = row['IDS'].decode('utf-8') if isinstance(row['IDS'], bytes) else str(row['IDS'])
                        names = [n.strip() for n in ids_str.split('|') if n.strip()]

                    result = {
                        'found': True,
                        'main_id': row['MAIN_ID'].decode('utf-8') if isinstance(row['MAIN_ID'], bytes) else str(row['MAIN_ID']),
                        'object_type': object_type,
                        'is_variable': True,
                        'names': names,
                        'ra': float(row['RA']) if 'RA' in row.colnames else None,
                        'dec': float(row['DEC']) if 'DEC' in row.colnames else None,
                        'num_matches': len(result_table)
                    }

                    logger.info(f"  SIMBAD: Found variable star - {result['main_id']} ({object_type})")
                    if result['num_matches'] > 1:
                        logger.info(f"         ({result['num_matches']} total objects in search radius)")

                    return result

            # No variable stars found, but other objects exist
            logger.debug(f"  SIMBAD: Found {len(result_table)} object(s) but none classified as variable")
            return {'found': True, 'is_variable': False, 'num_matches': len(result_table)}

        except Exception as e:
            logger.error(f"  SIMBAD: Error in coordinate search: {e}")
            return None

    def query_vsx(self, tic_number: int) -> Optional[Dict[str, Any]]:
        """
        Query AAVSO VSX (Variable Star Index) for a TIC

        Args:
            tic_number: TIC identifier

        Returns:
            Dictionary with VSX data or None if not found
        """
        try:
            # VSX API endpoint
            # Format: view=results.get&ident=TIC+123456
            url = f"{self.vsx_base_url}?view=results.get&ident=TIC+{tic_number}"

            logger.debug(f"  Querying VSX: {url}")

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # VSX returns HTML, check for actual variable entry
            content = response.text.lower()

            # Check for explicit "not found" messages
            if "no object found" in content or "no variables found" in content:
                logger.debug(f"  VSX: TIC {tic_number} not found")
                return {'found': False, 'tic_number': tic_number}

            # Better check: Look for actual variable data
            # Real VSX entries have an OID (object identifier) and VSX name
            import re

            # Look for VSX name pattern (e.g., "VSX J105515.3-735609")
            vsx_name_pattern = re.search(r'(VSX J[0-9+.-]+)', response.text, re.IGNORECASE)

            # Look for OID which only appears in actual results
            has_oid = 'oid' in content and ('oid=' in content or 'oid:' in content)

            # If no VSX name and no OID, it's just the search form (no results)
            if not vsx_name_pattern and not has_oid:
                logger.debug(f"  VSX: TIC {tic_number} not found (search form returned, no results)")
                return {'found': False, 'tic_number': tic_number}

            # Found actual variable entry
            result = {
                'found': True,
                'tic_number': tic_number,
                'url': url,
                'raw_html': content[:500]  # Store snippet for debugging
            }

            # Extract VSX name if found
            if vsx_name_pattern:
                result['name'] = vsx_name_pattern.group(1)
                logger.info(f"  VSX: Found {result['name']}")
            else:
                result['name'] = f"VSX entry for TIC {tic_number}"
                logger.info(f"  VSX: Found entry for TIC {tic_number}")

            return result

        except requests.exceptions.Timeout:
            logger.error(f"  VSX: Timeout querying TIC {tic_number}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"  VSX: Error querying TIC {tic_number}: {e}")
            return None

    def query_simbad(self, tic_number: int) -> Optional[Dict[str, Any]]:
        """
        Query SIMBAD for a TIC using astroquery

        Args:
            tic_number: TIC identifier

        Returns:
            Dictionary with SIMBAD data or None if not found
        """
        if not ASTROQUERY_AVAILABLE:
            logger.warning("  SIMBAD: astroquery not available, using fallback")
            return self._query_simbad_fallback(tic_number)

        try:
            # Query by TIC identifier
            identifier = f"TIC {tic_number}"
            logger.debug(f"  Querying SIMBAD: {identifier}")

            result_table = self.simbad.query_object(identifier)

            if result_table is None or len(result_table) == 0:
                logger.debug(f"  SIMBAD: TIC {tic_number} not found")
                return {'found': False, 'tic_number': tic_number}

            # Extract data from result
            row = result_table[0]

            # Get object type
            object_type = row['OTYPE'].decode('utf-8') if isinstance(row['OTYPE'], bytes) else str(row['OTYPE'])

            # Check if it's a variable star type
            variable_types = ['V*', 'EB', 'Pulsating', 'Eruptive', 'Cataclysmic']
            is_variable = any(vt in object_type for vt in variable_types)

            # Get all identifiers/names
            names = []
            if 'IDS' in row.colnames:
                ids_str = row['IDS'].decode('utf-8') if isinstance(row['IDS'], bytes) else str(row['IDS'])
                names = [n.strip() for n in ids_str.split('|') if n.strip()]

            result = {
                'found': True,
                'tic_number': tic_number,
                'main_id': row['MAIN_ID'].decode('utf-8') if isinstance(row['MAIN_ID'], bytes) else str(row['MAIN_ID']),
                'object_type': object_type,
                'is_variable': is_variable,
                'names': names,
                'ra': float(row['RA']) if 'RA' in row.colnames else None,
                'dec': float(row['DEC']) if 'DEC' in row.colnames else None
            }

            if is_variable:
                logger.info(f"  SIMBAD: Found variable star - {result['main_id']} ({object_type})")
            else:
                logger.debug(f"  SIMBAD: Found object but not classified as variable")

            return result

        except Exception as e:
            logger.error(f"  SIMBAD: Error querying TIC {tic_number}: {e}")
            return None

    def _query_simbad_fallback(self, tic_number: int) -> Optional[Dict[str, Any]]:
        """
        Fallback SIMBAD query using direct HTTP (if astroquery unavailable)

        Args:
            tic_number: TIC identifier

        Returns:
            Dictionary with basic SIMBAD data or None
        """
        try:
            # SIMBAD URL query format
            url = f"http://simbad.u-strasbg.fr/simbad/sim-id?Ident=TIC+{tic_number}&output.format=ASCII"

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            content = response.text

            # Check if object was found
            if "not found" in content.lower() or "no object" in content.lower():
                return {'found': False, 'tic_number': tic_number}

            # Basic parsing (not robust, but works for fallback)
            result = {
                'found': True,
                'tic_number': tic_number,
                'is_variable': 'V*' in content or 'variable' in content.lower(),
                'raw_data': content[:500]
            }

            return result

        except Exception as e:
            logger.error(f"  SIMBAD fallback: Error querying TIC {tic_number}: {e}")
            return None

    def _get_cached_result(self, tic_number: int) -> Optional[Dict[str, Any]]:
        """Get cached catalog result from Supabase"""
        if not settings.is_configured:
            return None

        try:
            cached = queries.get_catalog_cache(tic_number)

            if cached:
                # Check if cache is still valid (30 days)
                cached_at = datetime.fromisoformat(cached['cached_at'].replace('Z', '+00:00'))
                age = datetime.now(timezone.utc) - cached_at

                if age < timedelta(days=30):
                    # Cache is valid
                    cache_age_days = age.days
                    return {
                        'tic_number': tic_number,
                        'is_known_variable': cached['is_known_variable'],
                        'vsx_result': cached.get('vsx_result'),
                        'simbad_result': cached.get('simbad_result'),
                        'known_names': cached.get('known_names', []),
                        'variable_type': cached.get('known_type'),
                        'source': 'cache',
                        'cache_age': f"{cache_age_days} days",
                        'cached_at': cached['cached_at']
                    }

            return None

        except Exception as e:
            logger.error(f"Error retrieving cache for TIC {tic_number}: {e}")
            return None

    def _cache_result(self, result: Dict[str, Any]):
        """Cache catalog result in Supabase"""
        if not settings.is_configured:
            logger.debug("Supabase not configured, skipping cache")
            return

        try:
            tic_number = result['tic_number']
            queries.cache_catalog_data(
                tic_number=tic_number,
                vsx_result=result.get('vsx_result'),
                simbad_result=result.get('simbad_result'),
                is_known_variable=result['is_known_variable']
            )
            logger.debug(f"  Cached result for TIC {tic_number}")

        except Exception as e:
            logger.error(f"Error caching result for TIC {tic_number}: {e}")

    def is_new_discovery(self, tic_number: int) -> bool:
        """
        Simple check: is this a new discovery?

        Args:
            tic_number: TIC identifier

        Returns:
            True if NOT in catalogs (new discovery), False if known
        """
        result = self.check_tic(tic_number)
        return not result['is_known_variable']


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

    def _load_fetherolf_catalog(self) -> Optional[pd.DataFrame]:
        """
        Load Fetherolf et al. 2023 catalog

        Looks for catalog in data/validation/wang_catalog_real.csv

        Returns:
            DataFrame with TIC numbers, or None if not found
        """
        catalog_paths = [
            Path('data/validation/wang_catalog_real.csv'),
            Path('data/validation/fetherolf_catalog.csv'),
        ]

        for path in catalog_paths:
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    logger.info(f"✅ Loaded Fetherolf catalog: {len(df)} TESS variables from {path}")
                    return df
                except Exception as e:
                    logger.warning(f"Failed to load Fetherolf catalog from {path}: {e}")

        logger.warning("⚠️  Fetherolf catalog not found - will not check against published TESS variables")
        return None

    def check_fetherolf(self, tic_number: int) -> Optional[Dict[str, Any]]:
        """
        Check if TIC is in Fetherolf et al. 2023 catalog

        Args:
            tic_number: TIC identifier

        Returns:
            Dictionary with Fetherolf data if found, None otherwise
        """
        if self.fetherolf_catalog is None:
            return None

        try:
            match = self.fetherolf_catalog[self.fetherolf_catalog['TIC'] == tic_number]

            if len(match) > 0:
                row = match.iloc[0]
                result = {
                    'found': True,
                    'tic': int(row['TIC']),
                    'period': float(row['Period']),
                    'source': 'Fetherolf et al. 2023 (ApJS 268, 4)'
                }

                # Add optional fields if present
                if 'Tmag' in row:
                    result['magnitude'] = float(row['Tmag'])
                if 'Sector' in row:
                    result['sector'] = str(row['Sector'])
                if 'Type' in row:
                    result['variable_type'] = str(row['Type'])

                logger.info(f"  📚 Found in Fetherolf catalog (P={result['period']:.6f}d)")
                return result

            return None

        except Exception as e:
            logger.error(f"Error checking Fetherolf catalog: {e}")
            return None


def quick_check(tic_number: int) -> bool:
    """
    Quick check if TIC is a new discovery

    Args:
        tic_number: TIC identifier

    Returns:
        True if new discovery, False if known variable
    """
    checker = CatalogChecker()
    return checker.is_new_discovery(tic_number)


def detailed_check(tic_number: int) -> Dict[str, Any]:
    """
    Detailed catalog check with full results

    Args:
        tic_number: TIC identifier

    Returns:
        Dictionary with all catalog information
    """
    checker = CatalogChecker()
    return checker.check_tic(tic_number)
