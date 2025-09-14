# ==============================================================================
# Part 1: 原 Habitat (COCO) 分类 - 用于保持向后兼容性
# ==============================================================================

# 原始 COCO 类别 ID 到名称的映射
categories = {
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    61: 'toilet',
    62: 'tv',
    60: 'dining-table',
    69: 'oven',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    41: 'cup',
    39: 'bottle',
}

# 原始 COCO 类别 ID 到新索引 (0-14) 的映射
categories_id_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # potted plant
    59: 3,  # bed
    61: 4,  # toilet
    62: 5,  # tv
    60: 6,  # dining-table
    69: 7,  # oven
    71: 8,  # sink
    72: 9,  # refrigerator
    73: 10, # book
    74: 11, # clock
    75: 12, # vase
    41: 13, # cup
    39: 14, # bottle
}


# ==============================================================================
# Part 2: 统一的 AI2-THOR + Habitat 分类系统
# 该系统整合了两个数据集的物品，并为别名提供了映射
# ==============================================================================

name2index = {
    # === 家具 (Furniture) ===
    "chair": 0,                      # 椅子
    "ArmChair": 0,                   # 扶手椅 (别名)
    "sofa": 1,                       # 沙发
    "couch": 1,                      # 长沙发 (别名)
    "plant": 2,                      # 植物
    "potted plant": 2,               # 盆栽 (别名)
    "HousePlant": 2,                 # 室内植物 (别名)
    "bed": 3,                        # 床
    "DogBed": 3,                     # 狗床 (别名)
    "table": 6,                      # 桌子 (通用)
    "dining-table": 6,               # 餐桌 (别名)
    "DiningTable": 6,                # 餐桌 (别名)
    "Desk": 6,                       # 书桌 (别名)
    "CoffeeTable": 6,                # 茶几 (别名)
    "SideTable": 6,                  # 边桌 (别名)
    "CounterTop": 15,                # 柜台面
    "Dresser": 16,                   # 梳妆台
    "TVStand": 17,                   # 电视柜
    "Stool": 18,                     # 凳子
    "Footstool": 18,                 # 脚凳 (别名)
    "Ottoman": 19,                   # 脚凳，软凳
    "ShelvingUnit": 20,              # 置物架单元
    "LaundryHamper": 21,             # 洗衣篮
    "Safe": 22,                      # 保险箱

    # === 电器 (Electronics & Appliances) ===
    "toilet": 4,                     # 马桶
    "tv_monitor": 5,                 # 电视/显示器
    "tv": 5,                         # 电视 (别名)
    "Television": 5,                 # 电视 (别名)
    "oven": 7,                       # 烤箱
    "refrigerator": 9,               # 冰箱
    "Fridge": 9,                     # 冰箱 (别名)
    "Microwave": 23,                 # 微波炉
    "Toaster": 24,                   # 烤面包机
    "CoffeeMachine": 25,             # 咖啡机
    "Kettle": 26,                    # 水壶
    "WashingMachine": 27,            # 洗衣机
    "ClothesDryer": 28,              # 干衣机
    "VacuumCleaner": 29,             # 吸尘器
    "Laptop": 30,                    # 笔记本电脑
    "Desktop": 30,                   # 台式机 (别名)
    "RemoteControl": 31,             # 遥控器
    "CellPhone": 32,                 # 手机
    "Watch": 33,                     # 手表
    "LightSwitch": 34,               # 电灯开关
    "Faucet": 35,                    # 水龙头
    "ShowerHead": 36,                # 淋浴喷头

    # === 厨具与餐具 (Kitchenware & Tableware) ===
    "sink": 8,                       # 水槽
    "cup": 13,                       # 杯子
    "Mug": 13,                       # 马克杯 (别名)
    "bottle": 14,                    # 瓶子
    "Bottle": 14,                    # 瓶子 (别名)
    "WineBottle": 14,                # 酒瓶 (别名)
    "SprayBottle": 14,               # 喷雾瓶 (别名)
    "SoapBottle": 14,                # 皂液瓶 (别名)
    "Plate": 37,                     # 盘子
    "Bowl": 38,                      # 碗
    "Pan": 39,                       # 平底锅
    "Pot": 40,                       # 锅
    "Knife": 41,                     # 刀
    "ButterKnife": 41,               # 黄油刀 (别名)
    "Ladle": 42,                     # 勺子，长柄勺
    "Spatula": 43,                   # 锅铲
    "Spoon": 44,                     # 勺子
    "Fork": 45,                      # 叉子
    "DishSponge": 46,                # 洗碗海绵
    "PaperTowelRoll": 47,            # 纸巾卷
    "AluminumFoil": 48,              # 铝箔
    "SaltShaker": 49,                # 盐瓶
    "PepperShaker": 50,              # 胡椒瓶
    
    # === 杂项与装饰 (Misc & Decor) ===
    "book": 10,                      # 书
    "clock": 11,                     # 钟
    "AlarmClock": 11,                # 闹钟 (别名)
    "vase": 12,                      # 花瓶
    "Pillow": 51,                    # 枕头
    "Painting": 52,                  # 画
    "Window": 53,                    # 窗户
    "Blinds": 53,                    # 百叶窗 (别名)
    "Statue": 54,                    # 雕像
    "Doorway": 55,                   # 门口
    "DeskLamp": 56,                  # 台灯
    "FloorLamp": 56,                 # 落地灯 (别名)
    "Candle": 57,                    # 蜡烛
    "Box": 58,                       # 盒子
    "GarbageCan": 59,                # 垃圾桶
    "GarbageBag": 60,                # 垃圾袋
    "Newspaper": 61,                 # 报纸
    "Pen": 62,                       # 笔
    "Pencil": 62,                    # 铅笔 (别名)
    "CreditCard": 63,                # 信用卡
    "KeyChain": 64,                  # 钥匙链
    "CD": 65,                        # 光盘
    "RoomDecor": 66,                 # 房间装饰品
    "TableTopDecor": 66,             # 桌面装饰品 (别名)
    
    # === 消耗品与个人用品 (Consumables & Personal Items) ===
    "Apple": 67,                     # 苹果
    "Egg": 68,                       # 鸡蛋
    "Potato": 69,                    # 土豆
    "Lettuce": 70,                   # 生菜
    "Tomato": 71,                    # 番茄
    "Bread": 72,                     # 面包
    "Towel": 73,                     # 毛巾
    "HandTowel": 73,                 # 擦手巾 (别名)
    "TowelHolder": 74,               # 毛巾架
    "HandTowelHolder": 74,           # 擦手巾架 (别名)
    "ToiletPaper": 75,               # 卫生纸
    "ToiletPaperHanger": 76,         # 卫生纸架
    "ShowerCurtain": 77,             # 浴帘
    "SoapBar": 78,                   # 肥皂
    "ScrubBrush": 79,                # 擦洗刷
    "Plunger": 80,                   # 马桶搋
    "TissueBox": 81,                 # 纸巾盒
    "Boots": 82,                     # 靴子
    "Cloth": 83,                     # 布

    # === 运动与工具 (Sports & Tools) ===
    "TennisRacket": 84,              # 网球拍
    "BaseballBat": 85,               # 棒球棒
    "BasketBall": 86,                # 篮球
    "Dumbbell": 87,                  # 哑铃
    "WateringCan": 88,               # 洒水壶
    "Cart": 89,                      # 手推车
    "TeddyBear": 90                  # 泰迪熊
}

# 为了方便，可以创建一个反向映射
index2name = {v: k for k, v in name2index.items() if not isinstance(k, tuple)}