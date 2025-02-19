def srt_to_lines(file):
    with open(file, "r") as f:
        srt = f.read().strip().replace("\n\n", "\n").splitlines()
    entries = [srt[i : i + 3] for i in range(0, len(srt), 3)]
    lines = [line for idx, timestamp, line in entries]
    return lines


def srt_to_text(file):
    with open(file, "r") as f:
        srt = f.read().strip().replace("\n\n", "\n").splitlines()
    entries = [srt[i : i + 3] for i in range(0, len(srt), 3)]
    lines = [line for idx, timestamp, line in entries]
    text = "\n".join(lines)
    return text
