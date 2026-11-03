"""按COCO多边形生成二值遮罩；不训练模型，不覆盖原图。"""
from pathlib import Path
import argparse
import csv
import json
import math
import shutil
from PIL import Image, ImageDraw, ImageChops, ImageOps


def rasterize(size, annotations, category_id):
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    for annotation in annotations:
        if annotation['category_id'] != category_id:
            continue
        segments = annotation.get('segmentation')
        if not isinstance(segments, list):
            raise ValueError('目前只接受COCO多边形，不接受RLE')
        for polygon in segments:
            if len(polygon) < 6 or len(polygon) % 2:
                raise ValueError('无效多边形')
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in polygon):
                raise ValueError('多边形坐标无效')
            draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill=255)
    return mask


def pixels(mask):
    return mask.histogram()[255]


def build(annotation_file, image_dir, output):
    annotation_file, image_dir, output = map(Path, (annotation_file, image_dir, output))
    if output.exists():
        raise FileExistsError('输出目录已存在，未覆盖：' + str(output))
    data = json.loads(annotation_file.read_text(encoding='utf-8-sig'))
    categories = {item['name']: item['id'] for item in data['categories']}
    if set(categories) != {'fruit', 'invalid'}:
        raise ValueError('标签必须是fruit和invalid')
    # 先完整检查对应关系及尺寸，再创建输出目录。
    for item in data['images']:
        filename = item['file_name']
        if Path(filename).name != filename or '\\' in filename:
            raise ValueError('图片文件名必须为基础文件名')
        path = image_dir / filename
        with Image.open(path) as image:
            if image.size != (item['width'], item['height']):
                raise ValueError('尺寸不匹配：' + filename)
            image.verify()
    output.mkdir(parents=True)
    folders = ['images', 'fruit_masks', 'invalid_raw_masks', 'invalid_masks', 'valid_masks', 'overlays', 'valid_only']
    for name in folders:
        (output/name).mkdir()
    shutil.copy2(annotation_file, output/'annotations_original.json')
    rows = []
    overview = Image.new('RGB', (1200, len(data['images'])*300), '#f5f5f1')
    draw = ImageDraw.Draw(overview)
    for index, item in enumerate(data['images'], 1):
        key = f'ir_{index:03d}'
        source = image_dir/item['file_name']
        image = Image.open(source).convert('RGB')
        annotations = [a for a in data['annotations'] if a['image_id']==item['id']]
        if not any(a['category_id']==categories['fruit'] for a in annotations):
            raise ValueError('缺少fruit标注：' + item['file_name'])
        fruit = rasterize(image.size, annotations, categories['fruit'])
        invalid_raw = rasterize(image.size, annotations, categories['invalid'])
        invalid = ImageChops.darker(fruit, invalid_raw)
        valid = ImageChops.subtract(fruit, invalid)
        assert ImageChops.darker(valid, invalid).getbbox() is None
        assert ImageChops.add(valid, invalid).tobytes()==fruit.tobytes()
        fp, ip, vp = map(pixels, (fruit,invalid,valid))
        if fp == 0:
            raise ValueError('fruit区域为空')
        # 原图按字节复制，英文别名和原始文件名对应关系记录在CSV。
        shutil.copy2(source, output/'images'/(key+source.suffix.lower()))
        for name, mask in [('fruit_masks',fruit),('invalid_raw_masks',invalid_raw),('invalid_masks',invalid),('valid_masks',valid)]:
            mask.save(output/name/(key+'.png'))
        grey = Image.blend(image, Image.new('RGB',image.size,(115,115,115)), 0.8)
        overlay = Image.composite(grey,image,invalid)
        overlay.save(output/'overlays'/(key+'.png'))
        valid_only = Image.composite(image,Image.new('RGB',image.size,'black'),valid)
        valid_only.save(output/'valid_only'/(key+'.png'))
        row = dict(id=key,original_filename=item['file_name'],width=image.width,height=image.height,
                   fruit_pixels=fp,invalid_pixels=ip,valid_pixels=vp,
                   invalid_percent=round(100*ip/fp,2),valid_percent=round(100*vp/fp,2),
                   invalid_outside_fruit_pixels=pixels(invalid_raw)-ip)
        rows.append(row)
        for col,(im,title) in enumerate([(image,'Original'),(overlay,'Invalid = grey'),(valid_only,'Valid area only')]):
            thumb = ImageOps.contain(im,(380,260))
            x=col*400+(400-thumb.width)//2;y=(index-1)*300+30
            overview.paste(thumb,(x,y))
            draw.text((col*400+12,(index-1)*300+8),key+' | '+title,fill='#293029')
    with (output/'statistics.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    overview.save(output/'overview.jpg',quality=92)
    cards=[]
    for row in rows:
        key=row['id']
        cards.append(f'<section><h2>{key} · 无效 {row["invalid_percent"]}% · 有效 {row["valid_percent"]}%</h2><div class="grid">'+''.join(f'<figure><a href="{folder}/{key}.png" target="_blank"><img src="{folder}/{key}.png"></a><figcaption>{title}</figcaption></figure>' for folder,title in [('images','原图'),('overlays','灰色覆盖为无效区'),('valid_only','仅保留有效果面')])+'</div></section>')
    # 本批上传源文件均为PNG。
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>红外无效区检查</title><style>body{margin:25px;background:#f5f5f1;color:#293029;font:14px "Microsoft YaHei",sans-serif}h1{font-size:23px}h2{font-size:15px;font-weight:500}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}figure{margin:0}img{width:100%;height:330px;object-fit:contain;background:#e5e8df}figcaption{padding:7px;color:#6c7961}section{padding:14px;background:white;margin:18px 0;border:1px solid #dce0d5}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style><h1>红外有效区域检查</h1><p>依据人工多边形生成，未使用AI预测。灰色代表无效；点击图片打开大图。比例基于人工标注果面，不代表损伤比例。</p>'''+''.join(cards)+'</html>'
    (output/'查看效果.html').write_text(html,encoding='utf-8')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--json',required=True)
    parser.add_argument('--images',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.json,args.images,args.output),ensure_ascii=False,indent=2))
