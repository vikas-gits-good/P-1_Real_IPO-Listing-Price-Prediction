import os


class S3Sync:
    def sync(self, source: str = None, destination: str = None):
        command = f"aws s3 sync {source} {destination}"
        os.system(command)
