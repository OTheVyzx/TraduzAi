from server.jobs.artifacts import _zip_member_name
from server.models import Artifact


def test_zip_member_name_keeps_translated_images_under_folder():
    artifact = Artifact(kind="translated_image", filename="001.jpg")

    assert _zip_member_name(artifact) == "translated/001.jpg"


def test_zip_member_name_strips_untrusted_path_segments():
    artifact = Artifact(kind="translated_image", filename="../001.jpg")

    assert _zip_member_name(artifact) == "translated/001.jpg"
