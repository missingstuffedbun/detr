import os
import random
import pandas as pd
import json


SOURCE_PATH = "D:\\toUser_第二批V2\\给选手数据\\Data(新增训练数据)"
DEST_PATH = "D:\\Car"
os.chdir(DEST_PATH)
random.seed(10)

CLASSES = dict(zip(
    ["stone", "truck", "car", "cone", "lamp", "scraper",
     "bulldozer", "sprinkler", "shovel", "Person", "drill", "excavator"],
    range(1,13)
))

TXT_COL = [
    'type',             # 目标类型，分别是stone，truck，car，cone，lamp，scraper，bulldozer，sprinkler，shovel，Person，drill，excavator
    'truncated',        # 物体被截断程度，由（0 1 ）表示
    'occuluded',        # 物体被遮挡程度，由（0 1 2 3）表示
    'alpha',            # 物体的观察角度，取值范围为：-pi~pi（单位：rad），具体示意如图所示。
    'xmin', 'ymin', 'xmax', 'ymax', # 目标2D检测框位置，左上顶点和右下顶点的像素坐标
    'height', 'width', 'length',    # 在激光雷达坐标系下，3D目标尺寸：高、宽、长
    'cx', 'cy', 'cz',               # 在激光雷达坐标系下，目标3D框中心坐标：（x,y,z）
    'rotation_y'        # 目标朝向角：[-pi,pi],顺时针为正
]


def create_annos(dirs, tag='train'):
    for d in dirs:
        for f in os.listdir(os.path.join(SOURCE_PATH,d,"Image")):
            os.rename(os.path.join(SOURCE_PATH,d,"Image",f), os.path.join(DEST_PATH,tag,f))
        txts = [f for f in os.listdir(os.path.join(SOURCE_PATH,d,"Txt"))]
        os.chdir(os.path.join(SOURCE_PATH,d,"Txt"))
        dfs = []
        for f in txts:
            print(f)
            _df = pd.read_csv(f, header=None, sep=' ', error_bad_lines=False)
            if len(_df) < 1:
                continue
            _df.columns = TXT_COL
            _df['file_name'] = f
            dfs.append(_df)
        df = pd.concat(dfs)
        df = df.reset_index()
        df["id"] = df.index
        image_id_map = dict(zip(pd.unique(df.file_name), range(1, len(pd.unique(df.file_name))+1)))
        df["image_id"] = df.file_name.map(image_id_map)
        df['file_name'] = df['file_name'].str.replace(".txt", ".jpg")
        df["category_id"] = df["type"].map(CLASSES)
        df["iscrowd"] = 0
        df['bbox_width'] = df['xmax']-df['xmin']
        df['bbox_height'] = df['ymax']-df['ymin']
        df['bbox'] = df[['xmin','ymin','bbox_width','bbox_height']].to_numpy().tolist()
        df['poly1'] = df[['xmin','ymin']].to_numpy().tolist()
        df['poly2'] = df[['xmax','ymin']].to_numpy().tolist()
        df['poly3'] = df[['xmax','ymax']].to_numpy().tolist()
        df['poly4'] = df[['xmin','ymax']].to_numpy().tolist()
        df['poly'] = df[['poly1','poly2','poly3','poly4']].to_numpy().tolist()
        df['segmentation'] = df['poly'].apply(lambda x: list(x))
        df['area'] = df['bbox_width'] * df['bbox_height']
        annos = {}
        annos["annotations"] = df[["id","image_id","category_id","segmentation","area","bbox","iscrowd"]].to_dict('records')
        img_df = df[['file_name','height','width','image_id']]
        img_df.rename(columns={'image_id':'id'}, inplace = True)
        img_df.drop_duplicates(inplace=True)
        annos["images"] = img_df.to_dict('records')
        with open(os.path.join(DEST_PATH, "custom_{}.json".format(tag)), "w") as outfile:
            json.dump(annos, outfile)

if __name__ == '__main__':
    groups = [d for d in os.listdir(SOURCE_PATH)]
    test_d = random.choices(groups, k=3)
    train_d = [d for d in groups if d not in test_d]
    # create_annos(test_d, 'test')
    create_annos(train_d, 'train')