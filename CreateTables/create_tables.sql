create table `1006`
(
    CSSJ timestamp(6) null comment '卫星时间码',
    1006_14 varchar(32) null comment '太阳阵输入电流',
    1006_16 varchar(32) null comment '分流域电压',
    1006_17 varchar(32) null comment '锂电池A主1充电电流',
    1006_18 varchar(32) null comment '锂电池A主2充电电流',
    1006_19 varchar(32) null comment '锂电池B主1充电电流',
    1006_20 varchar(32) null comment '锂电池B主2充电电流',
    1006_23 varchar(32) null comment '锂电池A放电电流',
    1006_24 varchar(32) null comment '锂电池B放电电流',
    1006_25 varchar(32) null comment '锂电池A单体1电压',
    1006_26 varchar(32) null comment '锂电池A单体2电压',
    1006_27 varchar(32) null comment '锂电池A单体3电压',
    1006_28 varchar(32) null comment '锂电池A单体4电压',
    1006_29 varchar(32) null comment '锂电池A单体5电压',
    1006_30 varchar(32) null comment '锂电池A单体6电压',
    1006_31 varchar(32) null comment '锂电池B单体1电压',
    1006_32 varchar(32) null comment '锂电池B单体2电压',
    1006_33 varchar(32) null comment '锂电池B单体3电压',
    1006_34 varchar(32) null comment '锂电池B单体4电压',
    1006_35 varchar(32) null comment '锂电池B单体5电压',
    1006_36 varchar(32) null comment '锂电池B单体6电压',
    1006_46 varchar(32) null comment '帆板1展开状态',
    1006_47 varchar(32) null comment '帆板2展开状态'
);


create table `1018`
(
    CSSJ timestamp(6) null comment '卫星时间码',
    1018_12 varchar(32) null comment '锂电池A温度1',
    1018_13 varchar(32) null comment '锂电池A温度2',
    1018_14 varchar(32) null comment '锂电池B温度1',
    1018_15 varchar(32) null comment '锂电池B温度2'
);


create table `101F`
(
    CSSJ timestamp(6) null comment '卫星时间码',
    101F_46 varchar(32) null comment '太阳敏感器底板温度_RM24',
    101F_47 varchar(32) null comment '太阳敏感器底板温度_RM25',
    101F_48 varchar(32) null comment '太阳敏感器顶板温度_RM26'
);


