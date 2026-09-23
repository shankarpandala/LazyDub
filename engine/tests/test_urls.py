import pytest

from maata_engine.urls import URLError, parse_youtube_url


@pytest.mark.parametrize(
    "url,vid,start,short",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ", 0, False),
        ("youtube.com/watch?v=dQw4w9WgXcQ&t=1m30s", "dQw4w9WgXcQ", 90, False),
        ("https://youtu.be/dQw4w9WgXcQ?t=42", "dQw4w9WgXcQ", 42, False),
        ("https://www.youtube.com/shorts/abcdefghijk", "abcdefghijk", 0, True),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ?start=10", "dQw4w9WgXcQ", 10, False),
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ&t=1h2m3s", "dQw4w9WgXcQ", 3723, False),
    ],
)
def test_parse(url, vid, start, short):
    ref = parse_youtube_url(url)
    assert (ref.video_id, ref.start, ref.is_short) == (vid, start, short)


@pytest.mark.parametrize(
    "url",
    ["", "https://vimeo.com/123", "https://www.youtube.com/watch?v=short", "https://www.youtube.com/live/dQw4w9WgXcQ", "https://www.youtube.com/playlist?list=PL1"],
)
def test_rejects(url):
    with pytest.raises(URLError):
        parse_youtube_url(url)
