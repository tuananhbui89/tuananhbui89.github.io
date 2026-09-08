from pathlib import Path
from PIL import Image, ImageDraw

folder = Path("/Users/tuananhbui/Personal/Gitpages/al-folio/tmp/pdfs")
pages = sorted(folder.glob("rendered-*.png"))
for sheet_index, start in enumerate(range(0, len(pages), 7), start=1):
    selected = pages[start:start + 7]
    thumbs = []
    for path in selected:
        im = Image.open(path).convert("RGB")
        im.thumbnail((760, 540))
        thumbs.append((path, im.copy()))
    width = 800
    row_h = 575
    sheet = Image.new("RGB", (width, row_h * len(thumbs)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (path, im) in enumerate(thumbs):
        y = row * row_h
        draw.text((12, y + 7), path.stem, fill="#12334A")
        sheet.paste(im, ((width - im.width) // 2, y + 28))
    sheet.save(folder / f"contact-{sheet_index}.png")
