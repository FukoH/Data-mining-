#!/usr/bin/python
# -*- coding: utf-8 -*
__author__ = 'alison'


"""
填充空值------根据每个指标的实际情况进行填充:

1、如果所有有指标空值，业务上都默认为0，可全部直接填充。将all_null = 1

2、如果各个指标的空值填充内容不同，all_null = 0,根据具体情况，修改代码

2.1) 第0类：a0  (填充0）
比如：lastmonth_real_fee等指标：从业务角度空值应填充成 0
代码示例： dict_fill.setdefault("a0",{})["columns"]=["bill_flag","lastmonth_real_fee"]
          dict_fill.setdefault("a0",{})["values"]=0

2.2) 第1类：a1  (填充2）
比如：终端成本类型(returned_fee_left)：正常情况下有两个值： 1 补贴券 2 非补贴券 ，从业务角度空值应填充成 2
代码示例： 设置填充指标：dict_fill.setdefault("a1",{})["columns"]=["returned_fee_left"]
          设置填充数值：dict_fill.setdefault("a1",{})["values"]=2

2.3) 第2类：a2  （填充平均值）
比如：其他指标(other_column1)：年龄等等指标填充为0，不合适，空值可填充平均值等
代码示例： 导入数据：    from audit_data import df
          设置填充指标：dict_fill.setdefault("a2",{})["columns"]=["other_column1"]
          设置填充数值：dict_fill.setdefault("a2",{})["values"]= mean

2.4) 第n类：an等等类似
比如：其他指标(other_column2)：根据情况填充
代码示例： 设置填充指标：dict_fill.setdefault("an",{})["columns"]=["other_column2"]
          设置填充数值：dict_fill.setdefault("an",{})["values"]=-999

"""

#设置填充字典
dict_fill= {}

#手动设置all_null值：
# 如果所有有指标空值，业务上都默认为0，可全部直接填充。将all_null = 1 ，null_columns_0值注释掉
all_null = 1
# 如果部分指标填充成0，部分指标填充成其它，则all_null = 0，并修改null_columns_0等于空值默认为0指标，null_columns_other1等于默认为其它指标，比如默认为2.null_columns_other2填充均值等等
dict_fill.setdefault("a0",{})["columns"]=["bill_flag",
"lastmonth_real_fee",
"active_flag",
"fee",
"cur_mon_owe_fee",
"ctc_amt",
"base_times",
"roam_times",
"calling_cuc_times",
"calling_cmc_times",
"calling_ctc_times",
"noroam_times",
"call_toll_times",
"call_notoll_times",
"calling_ctc_duration",
"calling_cmc_duration",
"calling_cuc_duration",
"roam_duration",
"noroam_duration",
"call_toll_duration",
"base_duration",
"toll_duration",
"loacl_duration",
"loacl_notoll_duration",
"loacl_calling_duration",
"calling_billing_duration",
"gj_roam_calling_bill_dur",
"gj_roam_called_bill_dur",
"gn_bill_dura",
"gj_bill_dura",
"gat_roam_bill_dura",
"loacl_notoll_bill_dur",
"p2p_mo_times",
"gj_roam_sms_tims",
"prof_sms_num",
"mms_times",
"ix_base_duration",
"ix_roam_duration",
"ix_kbytes",
"ix_mo_base_kbytes",
"ix_mo_roam_kbytes",
"ix_mt_base_kbytes",
"ix_mt_roam_kbytes",
"g2_dura",
"g3_dura",
"g2_flux",
"g3_flux",
"ix_times",
"ix_base_kbytes",
"ix_roam_kbytes",
"gj_roam_ix_flux",
"g4_dura",
"g4_flux",
"mtd_userflow_wx",
"mtd_userflow_yx"]
dict_fill.setdefault("a0",{})["values"]=0


"""
dict_fill.setdefault("a1",{})["columns"]=["returned_fee_left"]
dict_fill.setdefault("a1",{})["values"]=2

from audit_data import df
dict_fill.setdefault("a2",{})["columns"]=["other_column1"]
dict_fill.setdefault("a2",{})["values"]=df.loc[:, dict_fill["a2"]['columns']].mean()
"""
