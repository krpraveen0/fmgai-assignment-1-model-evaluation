"""Download and verify the official SEN12MS summer archives."""  # ADDED

import argparse  # ADDED
from concurrent.futures import ThreadPoolExecutor  # ADDED
import hashlib  # ADDED
import json  # ADDED
from pathlib import Path  # ADDED
import shutil  # ADDED
import subprocess  # ADDED
import tarfile  # ADDED
import tempfile  # ADDED
import time  # ADDED

SOURCE = "https://dataserv.ub.tum.de/s/m1474000"  # ADDED
CHECKSUMS = {  # ADDED
    "ROIs1868_summer_s1.tar.gz": "faa9489b2d22e763c5166b4bc0792402fe7b6fb32a14f00888e1940b5ede9bffc2fc911dc12e26cd3c33a0f76f297e127d0fdc0f58c719125fe99697b9d8a47a",  # ADDED
    "ROIs1868_summer_s2.tar.gz": "24e52a64ef25c3676b8113a578f3f492dee7667c157fda941ef83d91100443b4682b9617f7d7284667fcc9b38ce4a265caf296df400db4988b9a7cd7e84ee141",  # ADDED
}  # ADDED


def digest(path):  # ADDED
    value = hashlib.sha512()  # ADDED
    with path.open("rb") as stream:  # ADDED
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):  # ADDED
            value.update(chunk)  # ADDED
    return value.hexdigest()  # ADDED


def download(name, root):  # ADDED
    target = root / name  # ADDED
    if target.exists():  # ADDED
        if digest(target) != CHECKSUMS[name]:  # ADDED
            raise ValueError(f"Checksum mismatch: {target}")  # ADDED
        return target  # ADDED
    partial = root / (name + ".part")  # ADDED
    with tempfile.TemporaryDirectory() as tmp:  # ADDED
        cookie = Path(tmp) / "cookies"  # ADDED
        subprocess.run(  # ADDED
            [  # ADDED
                "curl",  # ADDED
                "--fail",  # ADDED
                "--silent",  # ADDED
                "--show-error",  # ADDED
                "--cookie-jar",  # ADDED
                str(cookie),  # ADDED
                SOURCE,  # ADDED
                "-o",  # ADDED
                str(Path(tmp) / "share.html"),  # ADDED
            ],  # ADDED
            check=True,  # ADDED
        )  # ADDED
        url = SOURCE + "/download?path=%2F&files=" + name  # ADDED
        command = [  # ADDED
            "curl",  # ADDED
            "--fail",  # ADDED
            "--location",  # ADDED
            "--silent",  # ADDED
            "--show-error",  # ADDED
            "--retry",  # ADDED
            "0",  # ADDED
            "--connect-timeout",  # ADDED
            "30",  # ADDED
            "--speed-time",  # ADDED
            "120",  # ADDED
            "--speed-limit",  # ADDED
            "1024",  # ADDED
            "--continue-at",  # ADDED
            "-",  # ADDED
            "--cookie",  # ADDED
            str(cookie),  # ADDED
            "--cookie-jar",  # ADDED
            str(cookie),  # ADDED
            url,  # ADDED
            "--output",  # ADDED
            str(partial),  # ADDED
        ]  # ADDED
        print("Downloading", name, flush=True)  # ADDED
        # ADDED: Start a fresh curl process so each retry resumes at the current size.
        for attempt in range(9):  # ADDED
            result = subprocess.run(command)  # ADDED
            if result.returncode == 0:  # ADDED
                break  # ADDED
            if attempt == 8:  # ADDED
                raise subprocess.CalledProcessError(result.returncode, command)  # ADDED
            print(f"Retrying {name} from {partial.stat().st_size if partial.exists() else 0} bytes", flush=True)  # ADDED
            time.sleep(min(2 ** attempt, 30))  # ADDED
    if digest(partial) != CHECKSUMS[name]:  # ADDED
        raise ValueError(f"Checksum mismatch; retained partial: {partial}")  # ADDED
    partial.rename(target)  # ADDED
    print("SHA-512 verified", name, flush=True)  # ADDED
    return target  # ADDED


def extract(archive, dest):  # ADDED
    marker = dest / (archive.name + ".extracted.json")  # ADDED
    if marker.exists():  # ADDED
        return  # ADDED
    with tarfile.open(archive, "r|gz") as stream:  # ADDED
        for member in stream:  # ADDED
            if not (member.isfile() or member.isdir()):  # ADDED
                raise ValueError("Archive contains links or special files")  # ADDED
            if (  # ADDED
                member.isfile()  # ADDED
                and shutil.disk_usage(dest).free < member.size + 2 * 2**30  # ADDED
            ):  # ADDED
                raise OSError("Less than 2 GiB reserve space")  # ADDED
            stream.extract(member, dest, filter="data")  # ADDED
    marker.write_text(  # ADDED
        json.dumps(  # ADDED
            {"archive": str(archive.resolve()), "sha512": CHECKSUMS[archive.name]}  # ADDED
        )  # ADDED
    )  # ADDED


def main():  # ADDED
    p = argparse.ArgumentParser()  # ADDED
    p.add_argument("--root", default="data")  # ADDED
    p.add_argument("--extract", action="store_true")  # ADDED
    args = p.parse_args()  # ADDED
    root = Path(args.root)  # ADDED
    archives = root / "archives"  # ADDED
    archives.mkdir(parents=True, exist_ok=True)  # ADDED
    if shutil.disk_usage(root).free < 80 * 2**30:  # ADDED
        raise OSError("Need at least 80 GiB available for the archive downloads")  # ADDED
    with ThreadPoolExecutor(max_workers=2) as pool:  # ADDED
        futures = [pool.submit(download, name, archives) for name in CHECKSUMS]  # ADDED
        while not all(f.done() for f in futures):  # ADDED
            progress = {  # ADDED
                name: (archives / (name + ".part")).stat().st_size  # ADDED
                if (archives / (name + ".part")).exists()  # ADDED
                else (  # ADDED
                    (archives / name).stat().st_size  # ADDED
                    if (archives / name).exists()  # ADDED
                    else 0  # ADDED
                )  # ADDED
                for name in CHECKSUMS  # ADDED
            }  # ADDED
            (root / "download_status.json").write_text(  # ADDED
                json.dumps(  # ADDED
                    {"source": SOURCE, "downloaded_bytes": progress, "complete": False},  # ADDED
                    indent=2,  # ADDED
                )  # ADDED
            )  # ADDED
            time.sleep(5)  # ADDED
        paths = [f.result() for f in futures]  # ADDED
    if args.extract:  # ADDED
        for path in paths:  # ADDED
            extract(path, root)  # ADDED
    (root / "download_status.json").write_text(  # ADDED
        json.dumps(  # ADDED
            {  # ADDED
                "source": SOURCE,  # ADDED
                "archives": [str(p) for p in paths],  # ADDED
                "checksums_verified": True,  # ADDED
                "extraction_requested": args.extract,  # ADDED
                "complete": True,  # ADDED
            },  # ADDED
            indent=2,  # ADDED
        )  # ADDED
    )  # ADDED


if __name__ == "__main__":  # ADDED
    main()  # ADDED
