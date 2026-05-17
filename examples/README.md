# Examples

## `sample.wav`

An 11-second clip from JFK's 1961 inaugural address. Public domain (US Government work). Sourced from [whisper.cpp samples](https://github.com/ggerganov/whisper.cpp/tree/master/samples).

```
"And so, my fellow Americans, ask not what your country can do for you,
 ask what you can do for your country."
```

## Smoke test

Verify your install works end-to-end:

```bash
stt examples/sample.wav --no-diarize -o /tmp/stt-test
diff -q /tmp/stt-test/sample_transcript/sample.transcript.txt \
        examples/expected_output/sample.transcript.txt
```

Or run with default diarization (will return one speaker for this monologue):

```bash
stt examples/sample.wav -o /tmp/stt-test
```

Expected output in [`expected_output/`](expected_output/):

- `sample.transcript.txt` — timestamped, single SPEAKER_00 line
- `sample.SPEAKER_00.txt` — per-speaker text
- `sample.srt` — subtitle format
- `sample.json` — structured (segments + merged turns)

For a multi-speaker test, point `stt` at any 2+ speaker audio file of your own.
