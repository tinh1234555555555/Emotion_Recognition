// Đối tượng ánh xạ tên model đầy đủ với tên ngắn gọn


// Thiết lập model mặc định khi trang được tải
document.addEventListener("DOMContentLoaded", function () {
  const modelChoiceElement = document.getElementById("model_choice");

  // Thêm các tùy chọn vào dropdown với tên đầy đủ
  for (const model in modelNames) {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model; // Hiển thị tên đầy đủ trong dropdown
    modelChoiceElement.appendChild(option);
  }

  modelChoiceElement.value = "naive_bayes"; // Đặt model mặc định

  // Xóa giá trị model đã lưu trong localStorage
  localStorage.removeItem("selectedModel");
});

// Lắng nghe sự kiện 'change' cho model
document
  .getElementById("model_choice")
  .addEventListener("change", async function () {
    const selectedModel = this.value;

    // Gửi yêu cầu đến server để thay đổi model
    try {
      const response = await fetch("/start_camera", {
        method: "POST",
        body: new URLSearchParams({ model_choice: selectedModel }),
      });

      const data = await response.json();
      if (data.status === "success") {
        console.log("Model đã được thay đổi thành công!");
      } else {
        console.error("Lỗi:", data.error);
      }
    } catch (error) {
      console.error("Error:", error);
    }
  });

// Đảm bảo rằng model được chọn là mặc định khi trang được tải
document
  .getElementById("startCamera")
  .addEventListener("click", async function () {
    const selectedModel = document.getElementById("model_choice").value;

    // Gửi yêu cầu đến server để bắt đầu camera với model mặc định
    const formData = new FormData();
    formData.append("model_choice", selectedModel);

    const response = await fetch("/start_camera", {
      method: "POST",
      body: formData,
    });

    // Cập nhật tên model đang sử dụng
    const currentModelDiv = document.getElementById("currentModel");
    currentModelDiv.textContent = `Model: ${modelNames[selectedModel]}`; // Hiển thị tên ngắn gọn
  });
