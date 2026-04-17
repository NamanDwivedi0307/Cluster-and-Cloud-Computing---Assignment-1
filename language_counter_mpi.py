import json
import sys
from collections import Counter
from mpi4py import MPI


def normalize_langs(value):
    """
    Convert a language field into a clean list of language codes.
    No merging is performed, e.g. 'en' and 'en-US' are treated separately.
    """
    if value is None:
        return []

    if isinstance(value, str):
        lang = value.strip()
        return [lang] if lang else []

    if isinstance(value, list):
        clean_langs = []
        for item in value:
            if isinstance(item, str):
                lang = item.strip()
                if lang:
                    clean_langs.append(lang)
        return clean_langs

    return []


def extract_mastodon_langs(obj):
    """
    Mastodon language field is expected under obj["doc"]["language"].
    Usually a string, but list is also tolerated.
    """
    doc = obj.get("doc", {})
    lang = doc.get("language", None)
    return normalize_langs(lang)


def extract_bluesky_langs(obj):
    """
    BlueSky language field is expected under obj["record"]["langs"].
    Usually a list, but string is also tolerated.
    """
    record = obj.get("record", {})
    langs = record.get("langs", None)
    return normalize_langs(langs)


def process_file(file_path, platform, rank, size):
    counts = Counter()

    stats = {
        "processed_lines": 0,
        "non_empty_lines": 0,
        "valid_json_lines": 0,
        "bad_json_lines": 0,
        "posts_with_no_language": 0,
        "counted_language_entries": 0,
    }

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            # Round-robin assignment across MPI ranks
            if line_num % size != rank:
                continue

            stats["processed_lines"] += 1
            line = line.strip()

            if not line:
                continue

            stats["non_empty_lines"] += 1

            try:
                obj = json.loads(line)
                stats["valid_json_lines"] += 1
            except json.JSONDecodeError:
                stats["bad_json_lines"] += 1
                continue

            if platform == "mastodon":
                langs = extract_mastodon_langs(obj)
            elif platform == "bluesky":
                langs = extract_bluesky_langs(obj)
            else:
                langs = []

            if not langs:
                stats["posts_with_no_language"] += 1
                continue

            for lang in langs:
                counts[lang] += 1
                stats["counted_language_entries"] += 1

    return counts, stats


def merge_stats(all_stats):
    merged = {
        "processed_lines": 0,
        "non_empty_lines": 0,
        "valid_json_lines": 0,
        "bad_json_lines": 0,
        "posts_with_no_language": 0,
        "counted_language_entries": 0,
    }

    for stats in all_stats:
        for key in merged:
            merged[key] += stats[key]

    return merged


def reduce_results(comm, local_counts, local_stats):
    gathered_counts = comm.gather(local_counts, root=0)
    gathered_stats = comm.gather(local_stats, root=0)

    final_counts = None
    final_stats = None

    if comm.Get_rank() == 0:
        final_counts = Counter()
        for c in gathered_counts:
            final_counts.update(c)
        final_stats = merge_stats(gathered_stats)

    return final_counts, final_stats


def print_platform_results(platform_name, file_path, counts, stats):
    print(f"\n===== {platform_name} =====")
    print(f"Input file: {file_path}")
    print("Top 10 languages:")

    for lang, count in counts.most_common(10):
        print(f"{lang}: {count}")

    print("\nStats:")
    print(f"Processed lines (across all ranks): {stats['processed_lines']}")
    print(f"Non-empty lines: {stats['non_empty_lines']}")
    print(f"Valid JSON lines: {stats['valid_json_lines']}")
    print(f"Bad JSON lines: {stats['bad_json_lines']}")
    print(f"Posts with no language: {stats['posts_with_no_language']}")
    print(f"Counted language entries: {stats['counted_language_entries']}")


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(sys.argv) != 3:
        if rank == 0:
            print("Usage: python language_counter_mpi.py <mastodon_file> <bluesky_file>")
        sys.exit(1)

    mastodon_file = sys.argv[1]
    bluesky_file = sys.argv[2]

    comm.Barrier()
    start_time = MPI.Wtime()

    # Process Mastodon
    mastodon_local_counts, mastodon_local_stats = process_file(
        mastodon_file, "mastodon", rank, size
    )
    mastodon_counts, mastodon_stats = reduce_results(
        comm, mastodon_local_counts, mastodon_local_stats
    )

    # Process BlueSky
    bluesky_local_counts, bluesky_local_stats = process_file(
        bluesky_file, "bluesky", rank, size
    )
    bluesky_counts, bluesky_stats = reduce_results(
        comm, bluesky_local_counts, bluesky_local_stats
    )

    comm.Barrier()
    end_time = MPI.Wtime()

    if rank == 0:
        elapsed = end_time - start_time

        print("=" * 60)
        print("Language Count Results")
        print(f"MPI ranks used: {size}")
        print(f"Total execution time: {elapsed:.4f} seconds")
        print("=" * 60)

        print_platform_results("Mastodon", mastodon_file, mastodon_counts, mastodon_stats)
        print_platform_results("BlueSky", bluesky_file, bluesky_counts, bluesky_stats)
