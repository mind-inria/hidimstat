import os

path_file = os.path.dirname(os.path.abspath(__file__))

import pytest
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import urllib.request
import json


def get_list_of_images():
    name_figure = []
    url = "https://api.github.com/repos/hidimstat/hidimstat.github.io/contents/dev/_images"
    headers = {"Accept": "application/vnd.github.v3+json"}

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        for item in data:
            if not ("thumb" in item["name"]):
                name_figure.append(item["name"])
    print(name_figure)
    return name_figure


@pytest.mark.parametrize(
    "name_figure",
    get_list_of_images(),
)
@pytest.mark.mpl_image_compare(
    style="default",
    baseline_dir="baseline_images",
    tolerance=5,  # tolerance should be reduce to 0
)
def test_example_figure_generated(name_figure):
    # Download the baseline image from the specified URL
    baseline_url = (
        "https://github.com/hidimstat/hidimstat.github.io/raw/main/dev/_images/"
        + name_figure
    )
    baseline_path = os.path.join(
        path_file + "/baseline_images/", "test_example_figure_generated_" + name_figure
    )
    urllib.request.urlretrieve(baseline_url, baseline_path)

    img = mpimg.imread(path_file + "/../../../docs/_build/html/_images/" + name_figure)
    fig = plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    plt.imshow(img)
    plt.axis("off")
    plt.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)
    return fig
