// main.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include "tensor.h"
#include <filesystem>
#include <cstring>

#include <windows.h>

extern void test1();
// 读取 ONNX 文件到 std::vector<char>
std::vector<char> LoadONNXModel(const std::string& model_path) {
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open ONNX model file: " + model_path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Failed to read ONNX model file: " + model_path);
    }

    return buffer;
}

int main() {
    try {
		std::vector<float> testData = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
		std::vector<char> testBytes(testData.size() * sizeof(float));
		std::memcpy(testBytes.data(), testData.data(), testBytes.size());

        DC::Tensor a = DC::Tensor::Create<float>({ 2,3 }, std::move(testBytes));
		std::cout << "Tensor test typeSize = " << DC::Type::getSize(DC::TensorMeta::TensorType::Float) << std::endl;

		// Dense passthrough -> explicit editable start -> sparse write should not lose original dense content.
		{
			std::vector<char> denseBytes(sizeof(float) * 6);
			std::memcpy(denseBytes.data(), testData.data(), denseBytes.size());

			DC::Tensor t = DC::Tensor::Create<float>();
			t.setDense(std::move(denseBytes), { 2, 3 });
			const auto beforeSpan = t.data<float>();
			std::vector<float> before(beforeSpan.begin(), beforeSpan.end());
			if (before != testData) {
				throw std::runtime_error("Dense passthrough data<float>() mismatch");
			}

            {
                auto r = t[0].trySet<std::vector<float>>(std::vector<float>{10.0f, 20.0f, 30.0f});
                if (!r) throw std::runtime_error("Chained scalar write failed");
            }
            // Multi-level chained indexing should work without excessive allocations and preserve semantics.
            {
                auto r = t[1][2].trySet<float>(99.0f);
                if (!r) throw std::runtime_error("Chained scalar write failed");
            }
            {
                auto scalarRes = t[1][2].tryItem<float>();
                if (!scalarRes) throw std::runtime_error("Chained scalar read failed");
                const float scalar = scalarRes.get();
                auto scalarVecSpan = t.data<float>();
                if (scalar != 99.0f) {
                    throw std::runtime_error("Chained scalar write/read mismatch");
                }
            }

			const auto afterSpan = t.data<float>();
			std::vector<float> after(afterSpan.begin(), afterSpan.end());
			std::vector<float> expected = { 10.0f, 20.0f, 30.0f, 4.0f, 5.0f, 99.0f };
			if (after != expected) {
				throw std::runtime_error("Editable materialization or write mismatch");
			}
		}

		// Explicit fill API should broadcast to current shape.
		{
			DC::Tensor t = DC::Tensor::Create<float>();
			t.setDense(std::vector<char>(sizeof(float) * 6, 0), { 2, 3 });
			t.fill(3.5f);
			const auto afterSpan = t.data<float>();
			std::vector<float> after(afterSpan.begin(), afterSpan.end());
			std::vector<float> expected(6, 3.5f);
			if (after != expected) {
				throw std::runtime_error("fill() broadcast mismatch");
			}
            {
                auto res = t.tryItem<float>();
                if (res) throw std::runtime_error("item() should not succeed on multi-element tensor");
            }
		}

		// 0-d scalar semantics: empty shape is a scalar, dense bytes size == typeSize.
		{
			DC::Tensor s = DC::Tensor::Create<float>();
			s.fill(7.0f);
			if (!s.shape().empty()) {
				throw std::runtime_error("0-d scalar fill() must keep empty shape");
			}
			if (s.item<float>() != 7.0f) {
				throw std::runtime_error("0-d scalar item() mismatch");
			}
		}

		// Sub-tensor scalar broadcast: assigning scalar to a row should fill the whole row.
		{
			std::vector<float> init = { 1, 2, 3, 4, 5, 6 };
			std::vector<char> denseBytes(sizeof(float) * init.size());
			std::memcpy(denseBytes.data(), init.data(), denseBytes.size());

			DC::Tensor t = DC::Tensor::Create<float>();
			t.setDense(std::move(denseBytes), { 2, 3 });
            {
                auto r = t[0].trySet<float>(9.0f);
                if (!r) throw std::runtime_error("Sub-tensor scalar broadcast write failed");
            }
            const auto afterSpan = t.data<float>();
            std::vector<float> after(afterSpan.begin(), afterSpan.end());
            std::vector<float> expected = { 9, 9, 9, 4, 5, 6 };
            if (after != expected) {
                throw std::runtime_error("Sub-tensor scalar broadcast mismatch");
            }
            {
                auto res = t[0].tryItem<float>();
                if (res) throw std::runtime_error("item() should not succeed on non-singleton sub-tensor view");
            }
		}

		// 1D convention and missing elements: missing reads should be zero-filled.
		{
			DC::Tensor t = DC::Tensor::Create<float>();
			t.setDense(std::vector<char>(sizeof(float) * 3, 0), { 3 });
			// write a single element, then read whole tensor back
            {
                auto r = t[1].trySet<float>(2.0f);
                if (!r) throw std::runtime_error("1D write failed");
            }
			const auto span = t.data<float>();
			std::vector<float> v(span.begin(), span.end());
			std::vector<float> expected = { 0.0f, 2.0f, 0.0f };
			if (v != expected) {
				throw std::runtime_error("1D or missing-element zero-fill mismatch");
			}
		}

		// Path overflow should throw fast.
		{
			DC::Tensor t = DC::Tensor::Create<float>();
			t.setDense(std::vector<char>(sizeof(float) * 6, 0), { 2, 3 });
            {
                auto rr = t[0][0][0].tryRank();
                if (rr) throw std::runtime_error("rank() should not succeed on path overflow");
            }
		}

		Sleep(100);
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Error: " << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        return -1;
    }
    return 0;
}
