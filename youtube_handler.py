"""
youtube_handler.py
------------------
Fetches top-level comments from a YouTube video using the
YouTube Data API v3 (google-api-python-client).
"""

import re
import logging
from typing import List, Optional, Tuple

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

_YT_VIDEO_ID_PATTERN = re.compile(
    r"(?:v=|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})"
)


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract the 11-character video ID from any common YouTube URL format.

    Supported formats
    -----------------
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID

    Returns None if no ID can be found.
    """
    match = _YT_VIDEO_ID_PATTERN.search(url)
    return match.group(1) if match else None


def get_video_metadata(youtube_client, video_id: str) -> dict:
    """
    Return basic metadata (title, channel, view count) for a video.
    Returns an empty dict on failure.
    """
    try:
        response = youtube_client.videos().list(
            part="snippet,statistics",
            id=video_id,
        ).execute()

        items = response.get("items", [])
        if not items:
            return {}

        snippet = items[0]["snippet"]
        stats   = items[0].get("statistics", {})
        return {
            "title":         snippet.get("title", "N/A"),
            "channel":       snippet.get("channelTitle", "N/A"),
            "view_count":    int(stats.get("viewCount", 0)),
            "like_count":    int(stats.get("likeCount", 0)),
            "comment_count": int(stats.get("commentCount", 0)),
        }
    except HttpError as exc:
        logger.warning("Could not fetch video metadata: %s", exc)
        return {}


# ── Main fetch function ───────────────────────────────────────────────────────

def fetch_youtube_comments(
    api_key: str,
    video_url: str,
    max_comments: int = 500,
) -> Tuple[List[str], dict]:
    """
    Fetch up to *max_comments* top-level comments from a YouTube video.

    Parameters
    ----------
    api_key : str
        A valid YouTube Data API v3 key.
    video_url : str
        Full YouTube video URL.
    max_comments : int
        Maximum number of comments to retrieve (default 500).
        The API returns up to 100 per page; this function paginates automatically.

    Returns
    -------
    comments : list[str]
        Plain-text comment bodies.
    metadata : dict
        Video title, channel, view / like / comment counts.

    Raises
    ------
    ValueError
        If the URL does not contain a recognisable video ID.
    PermissionError
        If the API key is invalid or comments are disabled for the video.
    RuntimeError
        For unexpected API errors.
    """
    # 1. Parse video ID
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError(
            f"Could not extract a YouTube video ID from: '{video_url}'"
        )

    # 2. Build the API client
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise YouTube API client: {exc}") from exc

    # 3. Fetch metadata (best-effort)
    metadata = get_video_metadata(youtube, video_id)

    # 4. Paginate through commentThreads
    comments: List[str] = []
    next_page_token: Optional[str] = None

    try:
        while len(comments) < max_comments:
            page_size = min(100, max_comments - len(comments))

            request_kwargs = dict(
                part="snippet",
                videoId=video_id,
                maxResults=page_size,
                textFormat="plainText",
                order="relevance",      # "time" | "relevance"
            )
            if next_page_token:
                request_kwargs["pageToken"] = next_page_token

            response = youtube.commentThreads().list(**request_kwargs).execute()

            for item in response.get("items", []):
                text = (
                    item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                )
                if text.strip():
                    comments.append(text.strip())

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break   # No more pages

    except HttpError as exc:
        error_reason = exc.error_details[0].get("reason", "") if exc.error_details else ""

        if exc.resp.status == 403:
            if "commentsDisabled" in str(exc):
                raise PermissionError(
                    "Comments are disabled for this video."
                ) from exc
            raise PermissionError(
                "API key is invalid or quota exceeded. "
                "Check your YouTube Data API v3 key."
            ) from exc

        if exc.resp.status == 404:
            raise ValueError(
                f"Video not found. Check the URL. (video_id={video_id})"
            ) from exc

        raise RuntimeError(f"YouTube API error: {exc}") from exc

    logger.info("Fetched %d comments for video %s", len(comments), video_id)
    return comments, metadata
