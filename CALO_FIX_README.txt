SỬA AI CALO NHẬN SAI

Nguyên nhân: Food-101 không có lớp cua nguyên con, nên model closed-set vẫn ép chọn một lớp khác.
Bản sửa dùng CLIP zero-shot với tập nhãn rộng hơn, có cua/tôm/cá/món Việt và dùng chung CLIP của AI tìm kiếm ảnh.

Cách áp dụng:
1. Copy toàn bộ nội dung patch vào thư mục project 5 AI hiện tại.
2. Chọn Replace khi Windows hỏi.
3. Ctrl+C backend cũ.
4. Chạy lại: uvicorn api.main:app --reload
5. React đang chạy sẽ tự reload; nếu không thì npm run dev.
