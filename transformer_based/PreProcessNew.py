import random


# Hàm này dùng để cắt 1 câu thành các câu con, mặc định là tách thành các câu đơn.
# Các câu được cách nhau bởi dấu chấm "."
def splitSentences(num_sentences_split=1):
    inputsDir = 'corpus'
    output_dir = 'corpus_out'
    txtVi = "data_vi_sentences.txt"
    txtBana = "data_bana_sentences.txt"
    txtViFile = open(inputsDir + '/' + txtVi, encoding='utf-8', errors='ignore').read()
    txtBanaFile = open(inputsDir + '/' + txtBana, encoding='utf-8', errors='ignore').read()

    splitsVi = txtViFile.split('\n')
    splitsBana = txtBanaFile.split('\n')
    num_rows = len(splitsVi)
    for i in range(num_rows):
        print('row is ', i)
        if len(splitsVi[i].strip()) == 0 or splitsVi[i].__contains__(".."):
            continue
        sentences_arr_vi = splitsVi[i].split('. ')
        sentences_arr_bana = splitsBana[i].split('. ')
        num_sentences = len(sentences_arr_vi)
        if num_sentences > 0 and num_sentences > num_sentences_split:
            index = 1
            split_sentence_bana = ""
            split_sentence_vi = ""

            while index <= num_sentences:
                if index % num_sentences_split == 0:
                    vi_each_sentence = sentences_arr_vi[index - 1].strip()
                    if len(vi_each_sentence) > 0 and vi_each_sentence != '\n':
                        split_sentence_vi += vi_each_sentence + '\n'
                    split_sentence_vi = split_sentence_vi.replace('\n', '. ').strip()
                    txtViFile += '\n' + split_sentence_vi
                    split_sentence_vi = ""

                    bana_each_sentence = sentences_arr_bana[index - 1].strip()
                    if len(bana_each_sentence) > 0 and bana_each_sentence != '\n':
                        split_sentence_bana += bana_each_sentence
                    split_sentence_bana = split_sentence_bana.replace('\n', '. ').strip()
                    txtBanaFile += '\n' + split_sentence_bana
                    split_sentence_bana = ""
                else:
                    vi_each_sentence = sentences_arr_vi[index - 1].strip()
                    if len(vi_each_sentence) > 0 and vi_each_sentence != '\n':
                        split_sentence_vi += vi_each_sentence + '\n'

                    bana_each_sentence = sentences_arr_bana[index - 1].strip()
                    if len(bana_each_sentence) > 0 and bana_each_sentence != '\n':
                        bana_each_sentence = bana_each_sentence.replace('.\n', '\n')
                        split_sentence_bana += bana_each_sentence
                index += 1

            if len(split_sentence_vi) > 0:
                split_sentence_vi = split_sentence_vi.replace('\n', '. ').strip()
                txtViFile += '\n' + split_sentence_vi
            if len(split_sentence_bana) > 0:
                split_sentence_bana = split_sentence_bana.replace('\n', '. ').strip()
                txtBanaFile += '\n' + split_sentence_bana

    txtViFile = txtViFile.replace(':.', ':').replace('..', '.')
    txtBanaFile = txtBanaFile.replace(':.', '.').replace('..', '.')

    with open(output_dir + '/lang-vi-spr.txt', 'w', encoding='utf-8') as f:
        f.write(txtViFile)
    with open(output_dir + '/lang-bana-spr.txt', 'w', encoding='utf-8') as f:
        f.write(txtBanaFile)


# Hàm này dùng để nhân dữ liệu, mặc định là 10
# Input: Tôi đi học -> times=2
# Output:   Tôi đi học (câu gốc)
#           Tôi đi học
#           Tôi đi học
def timesMultipleData(times=10, inputsDir="output", output_dir="output", txtVi="lang-vi-spr.txt", txtBana="lang-bana-spr.txt", txtOutVi="lang-vi-spr", txtOutBana="lang-bana-spr"):
    # inputsDir = 'output'
    # txtVi = "lang-vi-spr.txt"
    # txtBana = "lang-bana-spr.txt"
    txtViFile = open(inputsDir + '/' + txtVi, encoding='utf-8', errors='ignore').read()
    txtBanaFile = open(inputsDir + '/' + txtBana, encoding='utf-8', errors='ignore').read()

    splitsVi = txtViFile.split('\n')
    splitsBana = txtBanaFile.split('\n')
    num_rows = len(splitsVi)

    txtViFile += '\n'
    txtBanaFile += '\n'
    for i in range(num_rows):
        if len(splitsVi[i].strip()) == 0:
            continue
        for j in range(times):
            txtViFile += '\n' + splitsVi[i].strip()
            txtBanaFile += '\n' + splitsBana[i].strip()

    numStr = str(times)
    with open(output_dir + '/' + txtOutVi + '-x' + numStr + '.txt', 'w', encoding='utf-8') as f:
        f.write(txtViFile)
    with open(output_dir + '/' + txtOutBana + '-x' + numStr + '.txt', 'w', encoding='utf-8') as f:
        f.write(txtBanaFile)


# Hàm này xử lý thay thế các dấu nháy kép và nháy đơn
def handleQuoutes():
    inputsDir = 'new_output'
    output_dir = 'new_output_new'
    txtVi = "vi_2903.txt"
    txtBana = "bana_2903.txt"
    txtViFile = open(inputsDir + '/' + txtVi, encoding='utf-8', errors='ignore').read()
    txtBanaFile = open(inputsDir + '/' + txtBana, encoding='utf-8', errors='ignore').read()

    txtViFile = txtViFile.replace("“", "\"").replace("”", "\"").replace("‘", "'")
    txtBanaFile = txtBanaFile.replace("“", "\"").replace("”", "\"").replace("‘", "'")
    with open(output_dir + '/vi_2903.txt', 'w', encoding='utf-8') as f:
        f.write(txtViFile)
    with open(output_dir + '/bana_2903.txt', 'w', encoding='utf-8') as f:
        f.write(txtBanaFile)


# Hàm này xáo trộn thứ tự các câu trong tập dữ liệu,
# tuy nhiên vị trí mới của ngôn ngữ nguồn vs ngôn ngữ đích tương đương
# Input: 2 file tiếng Việt và Bana
# Output: Thứ tự các câu trong 2 file thay đổi,
#           nhưng vị trí mới của câu trong tiếng Việt giống với câu trong tiếng Bana tương ứng
def shuffleRandomSentences(inputDir="new_output", outputDir="new_output", txtVi="vi_2903_no_x.txt", txtBana="bana_2903_no_x.txt", txtViOut="vi_2903_no_x_shuffle", txtBanaOut="bana_2903_no_x_shuffle"):
    # inputsDir = 'new_output'
    # txtVi = "vi_2903_no_x.txt"
    # txtBana = "bana_2903_no_x.txt"
    txtViFile = open(inputDir + '/' + txtVi, encoding='utf-8', errors='ignore').read()
    txtBanaFile = open(inputDir + '/' + txtBana, encoding='utf-8', errors='ignore').read()

    splitsVi = txtViFile.split('\n')
    splitsBana = txtBanaFile.split('\n')
    num_rows = len(splitsVi)

    generalArr = list(zip(splitsVi, splitsBana))
    random.shuffle(generalArr)
    splitsVi, splitsBana = zip(*generalArr)

    txtViFileNew = ""
    txtBanaFileNew = ""
    splitsVi = list(splitsVi)
    splitsBana = list(splitsBana)

    for i in range(num_rows):
        txtViFileNew += splitsVi[i].strip() + "\n"
        txtBanaFileNew += splitsBana[i].strip() + "\n"

    with open(outputDir + '/' + txtViOut + '_shuffle.txt', 'w', encoding='utf-8') as f:
        f.write(txtViFileNew)
    with open(outputDir + '/' + txtBanaOut + '_shuffle.txt', 'w', encoding='utf-8') as f:
        f.write(txtBanaFileNew)


# Hàm này xử lý dấu chấm, nếu câu tiếng Việt có dấu chấm và câu tiếng Bana không có thì thêm tương ứng
# Nếu đã xử lý chỗ này thì có thể bỏ qua hàm này
def handleEndSentence(inputsDir='corpus', outputDir='new_output', txtVi="vi-2903.txt", txtBana="bana-2903.txt", outVi="vi_2903", outBana="bana_2903"):
    # inputsDir = 'corpus'
    # outputDir = 'new_output'
    # txtVi = "vi-2903.txt"
    # txtBana = "bana-2903.txt"
    txtViFile = open(inputsDir + '/' + txtVi, encoding='utf-8', errors='ignore').read()
    txtBanaFile = open(inputsDir + '/' + txtBana, encoding='utf-8', errors='ignore').read()

    splitsVi = txtViFile.split('\n')
    splitsBana = txtBanaFile.split('\n')
    num_rows = len(splitsVi)

    txtViFileNew = ""
    txtBanaFileNew = ""

    for i in range(num_rows):
        if splitsVi[i].endswith(".") and (not splitsBana[i].endswith(".")):
            txtViFileNew += splitsVi[i] + "\n"
            txtBanaFileNew += splitsBana[i] + "." + "\n"
        elif splitsBana[i].endswith(".") and (not splitsVi[i].endswith(".")):
            txtBanaFileNew += splitsBana[i] + "\n"
            txtViFileNew += splitsVi[i] + "." + "\n"
        elif splitsVi[i].strip() != "" and splitsBana[i].strip() != "":
            txtViFileNew += splitsVi[i] + "\n"
            txtBanaFileNew += splitsBana[i] + "\n"

    with open(outputDir + '/' + outVi + '.txt', 'w', encoding='utf-8') as f:
        f.write(txtViFileNew)
    with open(outputDir + '/' + outBana + '.txt', 'w', encoding='utf-8') as f:
        f.write(txtBanaFileNew)


# Hàm này chia tập dữ liệu train và test, mặc định là chia 7-3
def train_test_split(trainSplitRate=0.7, inputDir="new_output_1", outputDir="new_output_1", txtVi="vi_0904_full_shuffle", txtBana="bana_0904_full_shuffle"):
    txtViFile = open(inputDir + '/' + txtVi + '.txt', encoding='utf-8', errors='ignore').read()
    txtBanaFile = open(outputDir + '/' + txtBana + '.txt', encoding='utf-8', errors='ignore').read()

    splitsVi = txtViFile.split('\n')
    splitsBana = txtBanaFile.split('\n')
    num_rows = len(splitsVi)
    train_rows = round(num_rows * trainSplitRate)
    test_rows = num_rows - train_rows

    if train_rows < 0 or train_rows > num_rows or test_rows < 0 or test_rows > num_rows:
        return
    txtViFileTrain = ""
    txtBanaFileTrain = ""
    txtViFileTest = ""
    txtBanaFileTest = ""
    for i in range(num_rows):
        if i < train_rows:
            txtViFileTrain += splitsVi[i].strip() + "\n"
            txtBanaFileTrain += splitsBana[i].strip() + "\n"
        else:
            txtViFileTest += splitsVi[i].strip() + "\n"
            txtBanaFileTest += splitsBana[i].strip() + "\n"

    with open(outputDir + '/' + txtVi + '_train.txt', 'w', encoding='utf-8') as f:
        f.write(txtViFileTrain)
    with open(outputDir + '/' + txtBana + '_train.txt', 'w', encoding='utf-8') as f:
        f.write(txtBanaFileTrain)

    with open(outputDir + '/' + txtVi + '_test.txt', 'w', encoding='utf-8') as f:
        f.write(txtViFileTest)
    with open(outputDir + '/' + txtBana + '_test.txt', 'w', encoding='utf-8') as f:
        f.write(txtBanaFileTest)
