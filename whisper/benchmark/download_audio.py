import yt_dlp

URLS = ['https://www.youtube.com/watch?v=0u7tTptBo9I']

ydl_opts = {
    'format': 'm4a/bestaudio/best',
    'outtmpl': 'benchmark.m4a',
    'postprocessors': [{  # Extract audio using ffmpeg
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'm4a',
    }]
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    error_code = ydl.download(URLS)
