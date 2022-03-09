import os
import pandas as pd
import json

SIZE = 256
DEST_PATH = "/content/working"

def read_bbox(f):
  return open(f,'r').readlines()

def sarship_coco_label(t="train"):
  label_json = {}
  file_names = [f for f in os.path.join(DEST_PATH,"labels",t) if f.endswith(".txt")]
  df = pd.DataFrame(file_names, columns=['file_name'])
  df["id"] = df.index
  df["image_id"] = df.index
  df['bbox'] = df['file_name'].apply(read_bbox)
  df['file_name'] = df['file_name'].str.replace(".txt",".jpg")
  df['height'] = SIZE
  df['width'] = SIZE
  label_json["images"] = df[['id','file_name','height','width']].to_dict('records')
  label_json["categories"] = [{"id":0,"name":"ship","supercategory":"none"}]
  df = df.explode('bbox')
  df_annotations = df[['id','image_id','file_name','bbox']]
  df_annotations["id"] = df_annotations.index
  df_annotations["category_id"] = 0
  df_annotations["ignore"] = 0
  df_annotations["iscrowd"] = 0
  df_annotations["bbox"] = df_annotations["bbox"].str.replace('\n', '')
  label_json["annotations"] = df_annotations.to_dict('records')
  for i in label_json["annotations"]:
      xcenter, ycenter, width, height = i['bbox'].split(" ")[1:]
      xcenter = float(xcenter)*SIZE
      ycenter = float(ycenter)*SIZE
      width = float(width)*SIZE
      height = float(height)*SIZE
      xmin = float(xcenter)-width/2
      ymin = float(ycenter)-height/2
      xmax = float(xcenter)+width/2
      ymax = float(ycenter)+height/2
      i['bbox'] = [xmin, ymin, width, height]
      poly = [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]
      i['segmentation'] = list([poly])
      i['area'] = width*height
  with open(os.path.join(DEST_PATH,"custom_{}.json".format(t)), "w") as outfile:
      json.dump(label_json, outfile)

if __name__=='__main__':
  sarship_coco_label("train")
  sarship_coco_label("test")
  sarship_coco_label("val")
