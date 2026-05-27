"""
Scans the 64x64 HDF5 cache for corrupt or incomplete files and deletes them.
Run from the project root:

    python src/check_cache.py
    python src/check_cache.py --delete
"""
import argparse
import sys
from pathlib import Path

import h5py
from tqdm import tqdm


EXPECTED_FRAMES_SHAPE = (3, 600, 64, 64)
EXPECTED_PPG_LEN      = 600


def check_file(p: Path) -> str | None:
    """Return an error description if the file is bad, else None."""
    try:
        with h5py.File(p, 'r') as f:
            if 'frames' not in f or 'ppg' not in f:
                return 'missing dataset'
            if f['frames'].shape != EXPECTED_FRAMES_SHAPE:
                return f'wrong frames shape {f["frames"].shape}'
            if f['ppg'].shape[0] != EXPECTED_PPG_LEN:
                return f'wrong ppg length {f["ppg"].shape}'
    except Exception as e:
        return str(e)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', default=None,
                        help='Path to scamps_cache_64 (auto-detected if omitted)')
    parser.add_argument('--delete', action='store_true',
                        help='Delete corrupt files after listing them')
    args = parser.parse_args()

    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        here = Path(__file__).resolve().parent.parent
        cache_dir = here / 'rppg_dataset' / 'SCAMPS' / 'scamps_cache_64'

    if not cache_dir.exists():
        print(f'Cache dir not found: {cache_dir}')
        sys.exit(1)

    files = sorted(cache_dir.glob('*.h5'))
    print(f'Scanning {len(files)} files in {cache_dir}')

    corrupt = []
    for p in tqdm(files, unit='file'):
        err = check_file(p)
        if err:
            corrupt.append((p, err))

    print(f'\nCorrupt / incomplete: {len(corrupt)}')
    for p, err in corrupt:
        print(f'  {p.name}  —  {err}')

    if corrupt and args.delete:
        for p, _ in corrupt:
            p.unlink()
            print(f'  Deleted {p.name}')
        print(f'Deleted {len(corrupt)} files. Re-run 02_build_cache.ipynb to rebuild them.')
    elif corrupt:
        print('\nRe-run with --delete to remove them, then rebuild via 02_build_cache.ipynb.')
    else:
        print('All files OK.')


if __name__ == '__main__':
    main()
