import pandas as pd

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import numpy as np

from sklearn.metrics import accuracy_score

import times
import numpy.lib.recfunctions as rfn

path = 'data'


def read_annotation(annotation_path='data/test_annotations.csv',
                    data=None, n_class=2):
    annotations = pd.read_csv(annotation_path)
    # data_catalog = pd.read_csv(annotation_catalog_path)
    return Annotations(annotations, data, n_class)


class Annotations:
    def __init__(self, raw_annotations, data, n_class):
        self.class_label = {}
        if n_class == 3:
            self.class_label = {"no": 0,
                                "yes": 1,
                                "maybe": 2}
        elif n_class == 2:
            self.class_label = {"no": 0,
                                "yes": 1,
                                "maybe": 1}
        self.n_class = n_class
        raw_annotations_cleaned = raw_annotations[raw_annotations['descriptions'].notnull()].copy()
        self.raw_annotations = raw_annotations_cleaned

        # self.annotation_catalog = annotation_catalog
        self.HUP_IDs = list(self.raw_annotations['HUP_ID'].unique())
        self.data = data
        self.annotation_dict = {}

        self.raters = list(self.raw_annotations['dataset'].unique())

        self.type_annot = {}
        self.time_annot = {}
        self.process_annotations()
        self.raters = list(self.annotation_dict.keys())
        self.annotations = pd.concat([self.annotation_dict[key] for key in self.raters])

    def process_annotations(self):
        self.raw_annotations.rename(columns={self.raw_annotations.columns[0]: "annotID"}, inplace=True)
        self.convert_time()

        for rater in self.raters:
            rater_annotation = self.raw_annotations[self.raw_annotations['dataset'].str.contains(rater)]
            self.type_annot[rater] = rater_annotation[rater_annotation['layerName'].str.contains("isSeizure?")]
            self.time_annot[rater] = rater_annotation[rater_annotation['layerName'].str.contains("SeizureTimes")]
            self.clean_type_description(rater)
            self.clean_channel_description(rater)
            self.make_cleaned_annotation(rater)

    def convert_time(self):
        start_time = []
        end_time = []
        annot_time_start = self.raw_annotations['annots_1']
        annot_time_end = self.raw_annotations['annots_2']
        start_time = [times.timestring_to_timestamp_annotation(st) for st in annot_time_start]
        end_time = [times.timestring_to_timestamp_annotation(et) for et in annot_time_end]

        assert len(start_time) == len(annot_time_start)
        assert len(start_time) == len(end_time)
        self.raw_annotations['Start Time'] = start_time
        self.raw_annotations['End Time'] = end_time

    # def clean_type_description(self, rater):
    #     annot_class = np.empty(len(self.type_annot[rater]))
    #     type_description = list(self.type_annot[rater]['descriptions'])
    #     for index, type_string in enumerate(type_description):
    #         tokens = nltk.tokenize.word_tokenize(str(type_string).lower())
    #         token_dict_keys = list(self.class_label.keys())
    #         elements = []
    #         for tk in tokens:
    #             if tk in token_dict_keys:
    #                 elements.append(self.class_label[tk])
    #         if len(elements) == 1:
    #             annot_class[index] = elements[0]
    #         else:
    #             annot_class[index] = 3
    #     temp_df = self.type_annot[rater].copy()
    #     temp_df['label'] = annot_class.astype(int)
    #     self.type_annot[rater] = temp_df

    def clean_type_description(self, rater):
        annot_class = np.empty(len(self.type_annot[rater]))
        type_description = list(self.type_annot[rater]['descriptions'])

        label_yes = self.type_annot[rater]['yes']
        label_no = self.type_annot[rater]['no']
        label_maybe = self.type_annot[rater]['maybe']
        to_drop = []
        if self.n_class == 2:
            annot_class = label_yes.copy()
        else:
            annot_class = label_yes.copy()
            test_ind = label_maybe.index[label_maybe == 1].tolist()
            annot_class[test_ind] = 2
        temp_df = self.type_annot[rater].copy()
        temp_df['label'] = annot_class.astype(int)

        episode_ind = np.empty(len(self.type_annot[rater]))
        episode_start_ind = np.empty(len(self.type_annot[rater]))
        episode_end_ind = np.empty(len(self.type_annot[rater]))
        for row_ind in range(len(temp_df)):
            row = temp_df.iloc[row_ind]
            start_time = row['Start Time']
            end_time = row['End Time']
            patientID = row['HUP_ID']
            clip_ind_start = times.extact_timestamp_to_ind(start_time, self.data[patientID].catalog, 'start')
            clip_ind_end = times.extact_timestamp_to_ind(end_time, self.data[patientID].catalog, 'end')
            if clip_ind_end == None and clip_ind_start == None:
                to_drop.append(row_ind)
                break
            if clip_ind_end == clip_ind_start:
                episode_ind[row_ind] = clip_ind_start
                episode_start_ind[row_ind] = self.data[patientID].catalog.iloc[clip_ind_start]['Event Start idx']
                episode_end_ind[row_ind] = self.data[patientID].catalog.iloc[clip_ind_start]['Event End idx']
            elif clip_ind_end == None:
                episode_ind[row_ind] = clip_ind_start
                episode_start_ind[row_ind] = self.data[patientID].catalog.iloc[clip_ind_start]['Event Start idx']
                episode_end_ind[row_ind] = self.data[patientID].catalog.iloc[clip_ind_start]['Event End idx']
            elif clip_ind_start == None:
                episode_ind[row_ind] = clip_ind_end
                episode_start_ind[row_ind] = self.data[patientID].catalog.iloc[clip_ind_end]['Event Start idx']
                episode_end_ind[row_ind] = self.data[patientID].catalog.iloc[clip_ind_end]['Event End idx']

        # if rater == 'RNS_Annotations_JimGugger':
        #     temp_df

        temp_df['Episode_Index'] = episode_ind.astype(int)
        temp_df['Episode_Start_Index'] = episode_start_ind.astype(int)
        temp_df['Episode_End_Index'] = episode_end_ind.astype(int)

        temp_df = temp_df.drop(temp_df.index[to_drop])
        self.type_annot[rater] = temp_df

    def clean_channel_description(self, rater):
        catalog_index_list = []
        channel_code_list = []
        to_drop = []
        for index in range(len(self.time_annot[rater])):
            catalog_ind = None
            row = self.time_annot[rater].iloc[index]
            channel_description = row['descriptions']
            start_time = row['Start Time']
            end_time = row['End Time']
            patientID = row['HUP_ID']
            episode_start_time_series = \
                self.type_annot[rater][self.type_annot[rater]['HUP_ID'].str.contains(patientID)][
                    'Start Time']
            episode_end_time_series = self.type_annot[rater][self.type_annot[rater]['HUP_ID'].str.contains(patientID)][
                'End Time']
            data_catalog = self.data[patientID].catalog

            catalog_ind_start, _ = times.timestamp_to_ind(start_time, data_catalog)
            catalog_ind_end, _ = times.timestamp_to_ind(end_time, data_catalog)

            master_index_list = self.type_annot[rater][self.type_annot[rater]['HUP_ID'].str.contains(patientID)][
                'Episode_Index'].tolist()

            ind_1 = np.where(catalog_ind_start == master_index_list)[0]
            ind_2 = np.where(catalog_ind_end == master_index_list)[0]
            rt_catalog_ind = 0
            if len(ind_1) == 1 and len(ind_2) == 1:
                if ind_1[0] == ind_2[0]:
                    rt_catalog_ind = ind_1[0]
                else:
                    st_d1 = start_time - data_catalog["Start UTC Timestamp"][master_index_list[ind_1[0]]]
                    et_d1 = data_catalog["End UTC Timestamp"][master_index_list[ind_1[0]]] - end_time
                    st_d2 = start_time - data_catalog["Start UTC Timestamp"][master_index_list[ind_2[0]]]
                    et_d2 = data_catalog["End UTC Timestamp"][master_index_list[ind_2[0]]] - end_time
                    if st_d1 >= 0 and et_d1 >= 0:
                        rt_catalog_ind = ind_1[0]
                    else:
                        rt_catalog_ind = None
                        to_drop.append(index)
            elif len(ind_1) == 0 and len(ind_2) == 1:
                rt_catalog_ind = ind_2[0]
            elif len(ind_1) == 1 and len(ind_2) == 0:
                rt_catalog_ind = ind_1[0]
            elif len(ind_1) == 0 and len(ind_2) == 0:
                rt_catalog_ind = None
                to_drop.append(index)

            if rt_catalog_ind != None:
                catalog_ind = \
                    self.type_annot[rater][self.type_annot[rater]['HUP_ID'].str.contains(patientID)].iloc[
                        rt_catalog_ind][
                        'annotID']

                row = self.time_annot[rater].iloc[index]
                start_time = row['Start Time']
                end_time = row['End Time']
                episode_start_time = self.raw_annotations.loc[catalog_ind, 'Start Time']
                episode_end_time = self.raw_annotations.loc[catalog_ind, 'End Time']
                start_time = np.max((start_time, episode_start_time))
                end_time = np.min((end_time, episode_end_time))

                self.time_annot[rater].iat[index, self.time_annot[rater].columns.get_loc('Start Time')] = start_time
                self.time_annot[rater].iat[index, self.time_annot[rater].columns.get_loc('End Time')] = end_time

                try:
                    assert start_time - episode_start_time >= 0 and episode_end_time - end_time >= 0
                except:
                    print("error" + str(catalog_ind))
                #
                catalog_index_list.append(catalog_ind)
                class_code = self.type_annot[rater].loc[catalog_ind, 'label']

                ch1 = row['ch1']
                ch2 = row['ch2']
                ch3 = row['ch3']
                ch4 = row['ch4']

                ch = [ch1, ch2, ch3, ch4]
                channel_code_list.append(ch)

        channel_code_arr = np.array(channel_code_list)
        c = channel_code_arr.astype(int).tolist()
        channel_code = [''.join(str(e) for e in line) for line in c]
        temp_df = self.time_annot[rater].copy()
        temp_df = temp_df.drop(temp_df.index[to_drop])
        temp_df['Catalog Index'] = catalog_index_list
        temp_df['Channel Code'] = channel_code
        temp_df['Binary Channel Code'] = channel_code
        self.time_annot[rater] = temp_df

    def make_cleaned_annotation(self, rater):
        type_annot = self.type_annot[rater]
        time_annot = self.time_annot[rater]
        annot_df = pd.DataFrame()
        annot_df["Dataset"] = type_annot['dataset']
        annot_df["Annotation_Catalog_Index"] = type_annot.index
        annot_df["Patient_ID"] = type_annot['HUP_ID']
        annot_df["Alias_ID"] = type_annot['aliasID']
        annot_df["Episode_Start_Timestamp"] = type_annot['Start Time']
        annot_df["Episode_End_Timestamp"] = type_annot['End Time']
        annot_df["Episode_Start_UTC_Time"] = [times.timestamp_to_utctime(ts) for ts in list(type_annot['Start Time'])]
        annot_df["Episode_End_UTC_Time"] = [times.timestamp_to_utctime(ts) for ts in list(type_annot['End Time'])]
        annot_df["Episode_Index"] = type_annot['Episode_Index']
        annot_df["Episode_Start_Index"] = type_annot['Episode_Start_Index']
        annot_df["Episode_End_Index"] = type_annot['Episode_End_Index']
        annot_df["Annotation_Start_Timestamp"] = [list() for _ in range(len(annot_df))]
        annot_df["Annotation_End_Timestamp"] = [list() for _ in range(len(annot_df))]
        annot_df["Annotation_Start_UTC_Time"] = [list() for _ in range(len(annot_df))]
        annot_df["Annotation_End_UTC_Time"] = [list() for _ in range(len(annot_df))]
        annot_df["Annotation_Start_Index"] = [list() for _ in range(len(annot_df))]
        annot_df["Annotation_End_Index"] = [list() for _ in range(len(annot_df))]
        annot_df["Type_Description"] = type_annot['descriptions']
        annot_df["Class_Code"] = type_annot['label']
        annot_df["Annotation_Channel"] = [list() for _ in range(len(annot_df))]
        annot_df["Channel_Code"] = [list() for _ in range(len(annot_df))]
        annot_df["Binary_Channel_Code"] = ['0000' for _ in range(len(annot_df))]

        for index in range(len(time_annot)):
            row = self.time_annot[rater].iloc[index]
            catalog_ind = row['Catalog Index']
            annot_df_index = annot_df.loc[annot_df['Annotation_Catalog_Index'] == catalog_ind].index
            # ==========================================================================
            annot_df.loc[annot_df_index, 'Annotation_Start_Timestamp'].values[0].append(row['Start Time'])
            annot_df.loc[annot_df_index, 'Annotation_End_Timestamp'].values[0].append(row['End Time'])
            # ==========================================================================
            annot_df.loc[annot_df_index, 'Annotation_Start_UTC_Time'].values[0].append(
                times.timestamp_to_utctime(row['Start Time']))
            annot_df.loc[annot_df_index, 'Annotation_End_UTC_Time'].values[0].append(
                times.timestamp_to_utctime(row['End Time']))
            annot_df.loc[annot_df_index, 'Annotation_Start_Index'].values[0].append(
                times.timestamp_to_ind(row['Start Time'],
                                       self.data[row["HUP_ID"]].catalog, thres=0)[1])
            annot_df.loc[annot_df_index, 'Annotation_End_Index'].values[0].append(
                times.timestamp_to_ind(row['End Time'],
                                       self.data[row["HUP_ID"]].catalog, thres=0)[1] - 1)
            # ==========================================================================
            annot_df.loc[annot_df_index, 'Annotation_Channel'].values[0].append(row['descriptions'])
            annot_df.loc[annot_df_index, 'Channel_Code'].values[0].append(row['Channel Code'])
            annot_df.loc[annot_df_index, 'Binary_Channel_Code'] = row['Binary Channel Code']


        annot_df = self.error_checking(annot_df)

        if len(annot_df) != 0:
            self.annotation_dict[rater] = annot_df

    def error_checking(self, annot_df):
        to_drop_list = []
        for index in range(len(annot_df)):
            is_code_valid = False
            is_annot_valid = False
            row = annot_df.iloc[index]
            class_code = row['Class_Code']
            type_description = row['Type_Description']
            channel_description = row['Annotation_Channel']
            channel_code = row['Channel_Code']
            is_code_valid = class_code != 3
            if is_code_valid:
                if class_code == 0:
                    is_annot_valid = True
                if class_code == 1 or class_code == 2:
                    is_annot_valid = len(channel_description) != 0
                if any([c == '3' for code in channel_code for c in code]):
                    is_annot_valid = False

            if is_annot_valid == False or is_code_valid == False:
                to_drop_list.append(row["Annotation_Catalog_Index"])
        for catalog_index in to_drop_list:
            annot_df = annot_df.drop(annot_df.loc[annot_df['Annotation_Catalog_Index'] == catalog_index].index)

        return annot_df

    def annot_match(self, match_annotation='type'):
        pred_dict = {}
        catalog_dict = {}
        for teacher_key in self.annotation_dict:
            teacher_annotation = self.annotation_dict[teacher_key]
            pred_dict[teacher_key] = {}
            catalog_dict[teacher_key] = {}
            for student_key in self.annotation_dict:
                pred_dict[teacher_key][student_key] = []
                catalog_dict[teacher_key][student_key] = []
                student_annotation = self.annotation_dict[student_key]
                for i in range(len(teacher_annotation)):
                    teacher_row = teacher_annotation.iloc[i]
                    eps_start_time = teacher_row["Episode_Start_Timestamp"]
                    eps_end_time = teacher_row["Episode_End_Timestamp"]
                    patientID = teacher_row["Patient_ID"]
                    student_row = student_annotation.loc[
                        (student_annotation['Episode_Start_Timestamp'] == eps_start_time) & (
                                student_annotation['Episode_End_Timestamp'] == eps_end_time) & (
                                student_annotation['Patient_ID'] == patientID)]
                    if match_annotation == 'type':
                        if len(student_row == 1):
                            pred_dict[teacher_key][student_key].append(
                                [int(teacher_row['Class_Code']), int(student_row['Class_Code'])])
                            catalog_dict[teacher_key][student_key].append(
                                [int(teacher_row['Annotation_Catalog_Index']),
                                 int(student_row['Annotation_Catalog_Index'])])
                    if match_annotation == 'channel':
                        if len(student_row == 1) and teacher_row['Binary_Channel_Code'] != []:
                            pred_dict[teacher_key][student_key].append(
                                [teacher_row['Binary_Channel_Code'], student_row['Binary_Channel_Code'].item()])
                            catalog_dict[teacher_key][student_key].append(
                                [int(teacher_row['Annotation_Catalog_Index']),
                                 int(student_row['Annotation_Catalog_Index'])])
                    if match_annotation == 'time':
                        if len(student_row == 1) and teacher_row['Binary_Channel_Code'] != []:
                            pred_dict[teacher_key][student_key].append(
                                [(teacher_row['Annotation_Start_Index'], teacher_row['Annotation_End_Index']), (
                                    student_row['Annotation_Start_Index'].item(),
                                    student_row['Annotation_End_Index'].item())])
                            catalog_dict[teacher_key][student_key].append(
                                [int(teacher_row['Annotation_Catalog_Index']),
                                 int(student_row['Annotation_Catalog_Index'])])

        return pred_dict, catalog_dict


def combine_annot_index(annot, patient_list, seed):
    np.random.seed(seed=seed)
    annot_nonseizure = annot[annot['Class_Code'] == 0]
    annot_seizure = annot[annot['Class_Code'] == 1]
    clip_dict = {}
    for p in patient_list:
        seizure_start_index = np.array([])
        seizure_end_index = np.array([])
        nonseizure_start_index = np.array([])
        nonseizure_end_index = np.array([])
        start_index = annot_seizure[annot_seizure['Patient_ID'] == p]['Episode_Start_Index']
        end_index = annot_seizure[annot_seizure['Patient_ID'] == p]['Episode_End_Index']
        annot_start_list = annot_seizure[annot_seizure['Patient_ID'] == p]['Annotation_Start_Index']
        annot_end_list = annot_seizure[annot_seizure['Patient_ID'] == p]['Annotation_End_Index']
        for i, slel in enumerate(zip(annot_start_list, annot_end_list)):
            sl = slel[0]
            el = slel[1]
            annot_array = np.vstack((sl, el))

            seizure_start_index = np.hstack((seizure_start_index, annot_array[0, :]))
            seizure_end_index = np.hstack((seizure_end_index, annot_array[1, :]))

            nonseizure_start_index = np.hstack((nonseizure_start_index, start_index.iloc[i]))
            nonseizure_end_index = np.hstack((nonseizure_end_index, annot_array[0, 0]))

            nonseizure_start_index = np.hstack((nonseizure_start_index, annot_array[1, -1]))
            nonseizure_end_index = np.hstack((nonseizure_end_index, end_index.iloc[i]))
            if annot_array.shape[1] > 1:
                nonseizure_start_index = np.hstack((nonseizure_start_index, annot_array[0, 1:]))
                nonseizure_end_index = np.hstack((nonseizure_end_index, annot_array[1, :-1]))

        nonseizure_valid = np.where(nonseizure_end_index - nonseizure_start_index > 500)
        seizure_valid = np.where(seizure_end_index - seizure_start_index > 500)

        nonseizure_ind_arr = np.vstack(
            (nonseizure_start_index[nonseizure_valid], nonseizure_end_index[nonseizure_valid])).astype(int)
        start_index = annot_nonseizure[annot_nonseizure['Patient_ID'] == p]['Episode_Start_Index']
        end_index = annot_nonseizure[annot_nonseizure['Patient_ID'] == p]['Episode_End_Index']

        print(np.vstack((seizure_start_index[seizure_valid], seizure_end_index[seizure_valid])).astype(int).shape)
        valid = np.where(end_index - start_index > 500)
        nonseizure_ind_arr_eps = np.vstack((start_index.iloc[valid], end_index.iloc[valid])).astype(int)

        if len(valid[0]) and len(seizure_valid[0]) > 0:
            nonseizure_clip_temp = np.hstack((nonseizure_ind_arr, nonseizure_ind_arr_eps))
            seizure_clip_temp = np.vstack(
                (seizure_start_index[seizure_valid], seizure_end_index[seizure_valid])).astype(
                int)

            nonseizure_clip_label = np.zeros(nonseizure_clip_temp.shape[1]).astype(int)
            seizure_clip_label = np.ones(seizure_clip_temp.shape[1]).astype(int)

            seizure_clip = np.vstack((seizure_clip_temp, seizure_clip_label))
            non_seizure_clip = np.vstack((nonseizure_clip_temp, nonseizure_clip_label))

            combined_clip = np.hstack((seizure_clip, non_seizure_clip))

            shuffled_index = np.arange(combined_clip.shape[1])
            np.random.shuffle(shuffled_index)

            clip_dict[p] = combined_clip[:, shuffled_index]

    return clip_dict


def annot_type_accuracy(pred_dict):
    result_dict = {}
    result_arr = np.empty((len(pred_dict), len(pred_dict)))
    print(result_arr.shape)
    for i, teacher_key in enumerate(pred_dict):
        result_dict[teacher_key] = {}
        for j, student_key in enumerate(pred_dict[teacher_key]):
            pred_arr = np.array(pred_dict[teacher_key][student_key])
            if len(pred_arr)>0:
                acc = accuracy_score(pred_arr[:, 0], pred_arr[:, 1])
                result_dict[teacher_key][student_key] = acc
                result_arr[i, j] = acc

    return result_dict, result_arr


def annot_channel_accuracy(pred_dict):
    channel_result_dict = {}
    channel_result_arr = np.empty((len(pred_dict), len(pred_dict)), dtype=object)
    print(channel_result_arr.shape)
    for i, teacher_key in enumerate(pred_dict):
        channel_result_dict[teacher_key] = {}
        for j, student_key in enumerate(pred_dict[teacher_key]):
            pred_arr = pred_dict[teacher_key][student_key]
            acc = sum([1 for k in range(len(pred_arr)) if pred_arr[k][0] == pred_arr[k][1]]) / len(pred_arr)
            channel_result_dict[teacher_key][student_key] = acc
            channel_result_arr[i, j] = acc

    return channel_result_dict, channel_result_arr


def annot_time_overlap(pred_dict):
    result_dict = {}
    result_arr = np.empty((len(pred_dict), len(pred_dict)), dtype=object)
    for i, teacher_key in enumerate(pred_dict):
        result_dict[teacher_key] = {}
        for j, student_key in enumerate(pred_dict[teacher_key]):
            total_overlap_time = 0
            total_total_time = 0
            pred_arr = pred_dict[teacher_key][student_key]
            for pair in pred_arr:
                start_1_list = pair[0][0]
                end_1_list = pair[0][1]
                start_2_list = pair[1][0]
                end_2_list = pair[1][1]
                overlap_time = 0
                total_time = 0
                for l1 in range(len(start_1_list)):
                    for l2 in range(len(start_2_list)):
                        overlap_time += find_overlap(start_1_list[l1], end_1_list[l1], start_2_list[l2], end_2_list[l2])
                try:
                    total_time = max(max(end_1_list), max(end_2_list)) - min(min(start_1_list), min(start_2_list)) + 1
                except:
                    total_time = 0
                total_overlap_time += overlap_time
                total_total_time += total_time
            result_dict[teacher_key][student_key] = total_overlap_time / total_total_time
            result_arr[i, j] = total_overlap_time / total_total_time

    return result_dict, result_arr


def find_overlap(start_1, end_1, start_2, end_2):
    return max(0, min(end_1, end_2) - max(start_1, start_2) + 1)


def check_diff(predict_dict, catalog_dict, teacher_key="BrianLitt_RNS_Test_Dataset"):
    diff_cataglog_ind = []
    diff_predict = []
    for j, student_key in enumerate(predict_dict[teacher_key]):
        pred_arr = np.array(predict_dict[teacher_key][student_key])
        for k in range(len(pred_arr)):
            if pred_arr[k][0] != pred_arr[k][1]:
                diff_cataglog_ind.append(catalog_dict[teacher_key][student_key][k])
                diff_predict.append(predict_dict[teacher_key][student_key][k])
    return diff_predict, diff_cataglog_ind


def sort_overlap(predict_dict, catalog_dict, teacher_key="BrianLitt_RNS_Test_Dataset"):
    start_time_diff = []
    end_time_diff = []
    diff_catalog_ind = []
    for j, student_key in enumerate(predict_dict[teacher_key]):
        total_overlap_time = 0
        total_total_time = 0
        pred_arr = predict_dict[teacher_key][student_key]
        for k, pair in enumerate(pred_arr):
            try:
                start_1 = pair[0][0][0]
                end_1 = pair[0][1][0]
                start_2 = pair[1][0][0]
                end_2 = pair[1][1][0]
            except:
                continue
            start_time_diff.append(abs(start_1 - start_2))
            end_time_diff.append(abs(end_1 - end_2))
            diff_catalog_ind.append(catalog_dict[teacher_key][student_key][k])

    return start_time_diff, end_time_diff, diff_catalog_ind

def combine_annotations(patient_list, annot_seizure, annot_nonseizure):
    clip_dict = {}
    for p in patient_list:
        seizure_start_index = np.array([])
        seizure_end_index = np.array([])
        nonseizure_start_index = np.array([])
        nonseizure_end_index = np.array([])
        global_episode_index_seizure = np.array([])
        global_episode_index_nonseizure = np.array([])

        start_index = annot_seizure[annot_seizure['Patient_ID'] == p]['Episode_Start_Index']
        end_index = annot_seizure[annot_seizure['Patient_ID'] == p]['Episode_End_Index']
        annot_start_list = annot_seizure[annot_seizure['Patient_ID'] == p]['Annotation_Start_Index']
        annot_end_list = annot_seizure[annot_seizure['Patient_ID'] == p]['Annotation_End_Index']
        episode_index = annot_seizure[annot_seizure['Patient_ID'] == p]['Episode_Index']

        for i, slel in enumerate(zip(annot_start_list, annot_end_list, episode_index.index)):
            sl_order = np.argsort(slel[0])
            sl = np.array(slel[0])[sl_order]
            el = np.array(slel[1])[sl_order]
            ei = slel[2]

            annot_array = np.vstack((sl, el))
            seizure_start_index = np.hstack((seizure_start_index, annot_array[0, :]))
            seizure_end_index = np.hstack((seizure_end_index, annot_array[1, :]))

            nonseizure_start_index = np.hstack((nonseizure_start_index, start_index.iloc[i]))
            nonseizure_end_index = np.hstack((nonseizure_end_index, annot_array[0, 0]))

            nonseizure_start_index = np.hstack((nonseizure_start_index, annot_array[1, -1]))
            nonseizure_end_index = np.hstack((nonseizure_end_index, end_index.iloc[i]))

            if annot_array.shape[1] > 1:
                nonseizure_start_index = np.hstack((nonseizure_start_index, annot_array[1, :-1]))
                nonseizure_end_index = np.hstack((nonseizure_end_index, annot_array[0, 1:]))

            global_episode_index_seizure = np.hstack((global_episode_index_seizure,
                                                      np.repeat(ei, len(seizure_start_index) -
                                                                len(global_episode_index_seizure))))
            global_episode_index_nonseizure = np.hstack((global_episode_index_nonseizure,
                                                         np.repeat(ei, len(nonseizure_start_index) -
                                                                   len(global_episode_index_nonseizure))))

        assert len(global_episode_index_nonseizure) == len(nonseizure_start_index)
        assert len(global_episode_index_seizure) == len(seizure_start_index)

        start_index = annot_nonseizure[annot_nonseizure['Patient_ID'] == p]['Episode_Start_Index']
        end_index = annot_nonseizure[annot_nonseizure['Patient_ID'] == p]['Episode_End_Index']
        episode_index = start_index.index

        nonseizure_ind_arr = np.vstack(
            [nonseizure_start_index,
             nonseizure_end_index,
             global_episode_index_nonseizure]).astype(int)

        seizure_ind_arr = np.vstack(
            [seizure_start_index,
             seizure_end_index,
             global_episode_index_seizure]).astype(int)

        nonseizure_ind_arr_eps = np.vstack(
            [start_index,
             end_index,
             episode_index]).astype(int)

        nonseizure_clip_temp = np.hstack((nonseizure_ind_arr, nonseizure_ind_arr_eps))
        nonseizure_clip_label = np.zeros(nonseizure_clip_temp.shape[1]).astype(int)
        non_seizure_clip = np.vstack((nonseizure_clip_temp, nonseizure_clip_label))

        seizure_clip_temp = np.vstack(
            [seizure_start_index,
             seizure_end_index,
             global_episode_index_seizure]).astype(int)
        seizure_clip_label = np.ones(seizure_clip_temp.shape[1]).astype(int)
        seizure_clip = np.vstack((seizure_clip_temp, seizure_clip_label))

        combined_clip = np.hstack((seizure_clip, non_seizure_clip))

        valid = np.where((combined_clip[1] - combined_clip[0]) > 500)

        combined_clip = combined_clip[:, valid].squeeze()

        if combined_clip.shape[1] > 0:
            # shuffled_index = np.arange(combined_clip.shape[1])
            # np.random.shuffle(shuffled_index)
            # clip_dict[p] = combined_clip[:, shuffled_index]

            structured_array = rfn.unstructured_to_structured(combined_clip.T.astype(int),
                                                              np.dtype(
                                                                  [('start_index', 'int32'), ('end_index', 'int32'),
                                                                   ('episode_index', 'int32'),
                                                                   ('label', 'int32')]))

            clip_dict[p] = combined_clip.T[np.argsort(structured_array, order=['episode_index', 'start_index'])].T

    return clip_dict