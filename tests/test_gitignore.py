"""Check that input data and generated training artifacts are ignored."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class GitignoreTests(unittest.TestCase):
    def test_training_artifacts_excluded_and_code_files_retained(self):
        excluded=("data/movielens/extra/train.tsv","data/movielens/movies.dat",
                  "data/movielens/movielens.train.rating","data/yelp/cache.json",
                  "Profile-Pollution-Toolkit/data/attack/train.rating",
                  "Profile-Pollution-Toolkit/patches/random.json",
                  "Profile-Pollution-Toolkit/log/run/metrics.json","outputs/run.txt",
                  "NCF/models/best.pth","LightGCN/checkpoints/best.safetensors",
                  "nested/logs/run.log","__pycache__/helper.pyc",
                  ".env",".env.example",".ssh/id_ed25519","private.pem")
        retained=("README.md","data/README.md","data/paper_split_sha256.txt",
                  "data/movielens/.gitkeep","data/yelp/.gitkeep","data/amazon_book/.gitkeep",
                  "results/main_due.csv","results/pollution_overall_ranking.json",
                  "Profile-Pollution-Toolkit/meta/targets.json","scripts/run_profile_pollution.py")
        with tempfile.TemporaryDirectory(prefix="due-gitignore-test-")as folder:
            root=Path(folder)
            shutil.copy2(ROOT/".gitignore",root/".gitignore")
            subprocess.run(["git","init","--quiet",str(root)],check=True)
            for name in excluded+retained:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text("temporary test fixture\n")
            subprocess.run(["git","-C",str(root),"add","--all"],check=True)
            tracked=set(subprocess.check_output(["git","-C",str(root),"ls-files","-z"]).decode().split("\0"))
            self.assertFalse(set(excluded)&tracked)
            self.assertTrue(set(retained)<=tracked)


if __name__=="__main__":unittest.main()
