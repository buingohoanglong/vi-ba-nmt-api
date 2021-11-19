# 1. Huấn luyện
    
    #Chạy câu lệnh sau: 
    python3 train.py -load_weights weights -SGDR 1 -floyd -batchsize 512 -checkpoint 20 -epochs 5 -printevery 10 -max_strlen 100 -src_data 'data/vi-0203-no-augment-shuffle.txt' -trg_data 'data/bana-0203-no-augment-shuffle.txt' -src_lang vi -trg_lang en
    #Lưu ý: Nếu huấn luyện lại từ đầu, bỏ tham số weights ra
# 2. Dịch
    python3 translate.py -load_weights weights -src_lang vi -trg_lang en
    Lưu ý: Do dung lượng khá nặng nên không up lên git, mô hình đã được huấn luyện với dữ liệu hiện tại có thế sử dụng ở đây: https://drive.google.com/drive/folders/1hghsAbDj6ZSWzPJ7fJoasNfj1gvIqWol?usp=sharing
# 3. Colab
    Nếu chạy bằng colab, vui lòng chạy file run_train.ipynb để huấn luyện và file run_translate.ipynb để dịch.

# 4.
    Nếu muốn dịch từng câu nhập vào, vui lòng comment hàm randomizeTranslate ở file translate.py và mở comment cho phần lặp bên dưới trong hàm main.

Chi tiết các tham số tham khảo nguồn ở đây: https://github.com/SamLynnEvans/Transformer
