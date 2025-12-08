from audioset_strong_download import Downloader

labels = ["Aircraft", "Fixed-wing aircraft, airplane", "Aircraft engine"]

cookies = "/mnt/acoustserv/cookies.txt"
d = Downloader(
    root_path="/mnt/acoustserv/audioset_strong/eval",
    labels=labels,
    n_jobs=2,
    dataset_ver="strong",
    download_type="eval",
    copy_and_replicate=False,
    cookies=cookies,
)
d.download(format="wav", quality=10)

d2 = Downloader(
    root_path="/mnt/acoustserv/audioset_strong/train",
    labels=labels,
    n_jobs=2,
    dataset_ver="strong",
    download_type="train",
    copy_and_replicate=False,
    cookies=cookies,
)
d2.download(format="wav", quality=10)


