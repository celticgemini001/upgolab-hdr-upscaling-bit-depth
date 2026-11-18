# Upscaling HDR Photographs: 8-bit vs 16-bit Measured, Where Highlight Detail Dies

Measured on a real photograph: a 16-bit tone-mapped intermediate preserves ripple detail to 100% where an 8-bit one preserves 0%, but the 16-bit label does nothing if you still tone-map before you grade. Codes-per-stop falls to 0.43 at the top, and 8-bit has 19.5 stops of total capacity, not 14.

**Read it with the tables and the reproduction snippets:** <https://celticgemini001.github.io/upgolab-hdr-upscaling-bit-depth/>

## About this repo

This repository exists to serve one measured article. Every number in it came from
ffmpeg and numpy run on real files on this machine — the commands are in the page so
you can re-run them and get the same rows. Where the measurement contradicted the
hypothesis, the page says so.

## The tool behind the notes

<https://upgolab.com> — browser-based AI upscaler for photos and short video.

## Index

All the guides are cross-linked from the hub:
<https://celticgemini001.github.io/upgolab-tools/>
