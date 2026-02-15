#pragma once
#include <algorithm>
#include <memory>
#include <stdexcept>

#include "TensorMods.h"

namespace DC {
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// DC::Tensor
// 张量对象
// 栈占用：264 字节
// 
// 用于创建输入或输出张量的数据载体。
//
// 形状术语约定：
// - CurrentShape: 当前数据形状（TensorData::getCurrentShape）
// - View::path : 索引路径，不等于 shape
// 
// 	2026.2.16
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	
	class Tensor {
		using TensorType = TensorMeta::TensorType;

	public:
		class View {
		public:
			View(Tensor& tensor)
				: _tensor(&tensor) {
			}

			View(Tensor& tensor, std::vector<int64_t> path)
				: _tensor(&tensor) {
				for (auto idx : path) {
					appendNode(idx);
				}
			}

			View operator[](uint64_t index) const {
				View next(*_tensor);
				next._chain = cloneChain();
				next._tail = next._chain.get() ? next.findTail(next._chain.get()) : nullptr;
				next.appendNode(static_cast<int64_t>(index));
				return next;
			}

			template<typename T>
			View& operator=(const std::vector<T>& data) {
				_tensor->writeAt(materializePath(), data);
				return *this;
			}

			// Add scalar assignment support
			template<typename T>
			View& operator=(const T& data) {
				_tensor->writeScalarAt(materializePath(), data);
				return *this;
			}

			template<typename T>
			std::vector<T> read() const {
				return _tensor->readAt<T>(materializePath());
			}

			// Add scalar read support (implicit conversion)
			template<typename T>
			T item() const {
				return _tensor->readScalarAt<T>(materializePath());
			}

			// Implicit conversion operators for common types
			operator float() const { return item<float>(); }
			operator double() const { return item<double>(); }
			operator int() const { return item<int>(); }
			operator int64_t() const { return item<int64_t>(); }
			operator bool() const { return item<bool>(); }
			operator char() const { return item<char>(); }
			operator unsigned char() const { return item<unsigned char>(); }

			const std::vector<int64_t>& path() const {
				_cachedPath = materializePath();
				return _cachedPath;
			}

		private:
			struct Node {
				const Node* parent = nullptr;
				int64_t index = 0;
				std::unique_ptr<Node> child;
			};

			Tensor* _tensor = nullptr;
			std::unique_ptr<Node> _chain;
			Node* _tail = nullptr;
			mutable std::vector<int64_t> _cachedPath;

			void appendNode(int64_t index) {
				auto n = std::make_unique<Node>();
				n->parent = _tail;
				n->index = index;
				Node* raw = n.get();
				if (!_chain) {
					_chain = std::move(n);
				}
				else {
					_tail->child = std::move(n);
				}
				_tail = raw;
			}

			Node* findTail(Node* root) const {
				Node* cur = root;
				while (cur && cur->child) {
					cur = cur->child.get();
				}
				return cur;
			}

			std::unique_ptr<Node> cloneChain() const {
				if (!_chain) {
					return nullptr;
				}
				auto newRoot = std::make_unique<Node>();
				newRoot->parent = nullptr;
				newRoot->index = _chain->index;
				Node* prev = newRoot.get();
				for (const Node* src = _chain->child.get(); src; src = src->child.get()) {
					auto next = std::make_unique<Node>();
					next->parent = prev;
					next->index = src->index;
					prev->child = std::move(next);
					prev = prev->child.get();
				}
				return newRoot;
			}

			std::vector<int64_t> materializePath() const {
				std::vector<int64_t> path;
				const Node* cur = _tail;
				while (cur) {
					path.push_back(cur->index);
					cur = cur->parent;
				}
				std::reverse(path.begin(), path.end());
				return path;
			}
		};

		virtual ~Tensor() = default;
		Tensor() = default;

		Tensor(
			const TensorType& type,					// - 张量类型
			size_t typeSize = 0,					// - 类型字节数（TypeSize）
			const std::vector<int64_t>& shape = {},	// - 数据形状（CurrentShape）
			std::vector<char>&& data = {}			// - 张量数据
		) {
			TensorMeta::ensureTypeMap();
			_meta.type = type;
			_meta.typeSize = (typeSize > 0) ? typeSize : DC::Type::getSize(type);
			_data = TensorData(shape, _meta.typeSize, std::move(data));
		}

		template<typename T>
		static Tensor Create(
			const std::vector<int64_t>& shape = {},
			std::vector<char>&& data = {}
		) {
			TensorMeta::ensureTypeMap();
			Tensor tensor(
				DC::Type::getType<TensorType, T>(),
				sizeof(T),
				shape,
				std::move(data)
			);

			return tensor;
		}

		View operator[](uint64_t index) {
			return View(*this, { static_cast<int64_t>(index) });
		}

		View operator[](int index) {
			return View(*this, { static_cast<int64_t>(index) });
		}

		View at() {
			return View(*this);
		}

		// Scalar item access for whole tensor (0-dim or 1-dim size 1)
		template<typename T>
		T item() const {
			auto v = _data.read<T>({});
			if (v.empty()) {
				throw std::runtime_error("Tensor: empty read result for scalar conversion.");
			}
			return v[0];
		}

		// 重载赋值
		Tensor& operator=(const Tensor& other) {
			_meta = other._meta;
			_data = other._data;
			return *this;
		}
		// 移动构造
		Tensor(Tensor&& other) noexcept {
			move_from(std::move(other));
		}

		// 移动赋值运算符
		Tensor& operator=(Tensor&& other) noexcept {
			if (this != &other) {
				move_from(std::move(other));
			}
			return *this;
		}

		// 在 Tensor 类的 public 部分添加拷贝构造函数
		Tensor(const Tensor& other) {
			_meta = other._meta;
			_data = other._data;
		}

		// 重载赋值
		// 给当前设定的维度写入数据
		template<typename T>
		Tensor& operator=(const std::vector<T>& data) {
			writeAt({}, data);
			return *this;
		}

		template<typename T>
		Tensor& operator=(const T& data) {
			writeAt({}, std::vector<T>{data});
			return *this;
		}

		const TensorType type() const { return _meta.type; }
		size_t typeSize() const { return _meta.typeSize; }

		// 获取一维化后的数据
		const std::vector<char> getData() const {
			return _data.getData<char>();
		}

		template<typename T>
		const std::vector<T> getData() const {
			return _data.getData<T>();
		}

		// 获取当前的动态形状（CurrentShape）
		std::vector<int64_t> getShape() const {
			return _data.getCurrentShape();
		}
		const std::vector<int64_t> shape() const { return _data.getCurrentShape(); }

		// 获取一维化后的原始字节数据（连续内存）。
		// 适合推理后端直接引用，避免每帧构造临时 vector<T>。
		const std::vector<char>& getBytes() const {
			return _data.getBytes();
		}

		// 直接设置为稠密（连续）数据。
		// bytes 为按 type 的字节序列（与 shape 对应）。
		Tensor& setDense(std::vector<char>&& bytes, const std::vector<int64_t>& shape) {
			_data.setDenseBytes(shape, _meta.typeSize, std::move(bytes));
			return *this;
		}

	private:
		friend class View;

		TensorMeta _meta;
		TensorData _data;

		template<typename T>
		void writeAt(const std::vector<int64_t>& path, const std::vector<T>& data) {
			_data.ensureEditable();
			_data.writeBitcast(path, _meta.typeSize, data);
		}

		template<typename T>
		void writeScalarAt(const std::vector<int64_t>& path, const T& data) {
			_data.ensureEditable();
			const auto shape = _data.getCurrentShape();
			if (!shape.empty() && path.size() == shape.size()) {
				const int64_t elementIndex = path.back();
				const int64_t blockSize = shape.back();
				if (elementIndex < 0 || elementIndex >= blockSize) {
					throw std::runtime_error("Tensor: scalar index out of range.");
				}
				std::vector<int64_t> blockPath(path.begin(), path.end() - 1);
				auto block = _data.read<T>(blockPath);
				if (block.empty()) {
					block.assign(static_cast<size_t>(blockSize), T{});
				}
				else if (static_cast<int64_t>(block.size()) < blockSize) {
					block.resize(static_cast<size_t>(blockSize), T{});
				}
				block[static_cast<size_t>(elementIndex)] = data;
				_data.writeBitcast(std::move(blockPath), _meta.typeSize, block);
				return;
			}
			_data.writeBitcast(path, _meta.typeSize, std::vector<T>{data});
		}

		template<typename T>
		std::vector<T> readAt(const std::vector<int64_t>& path) const {
			return _data.read<T>(path);
		}

		template<typename T>
		T readScalarAt(const std::vector<int64_t>& path) const {
			const auto shape = _data.getCurrentShape();
			if (!shape.empty() && path.size() == shape.size()) {
				const int64_t elementIndex = path.back();
				const int64_t blockSize = shape.back();
				if (elementIndex < 0 || elementIndex >= blockSize) {
					throw std::runtime_error("Tensor: scalar index out of range.");
				}
				std::vector<int64_t> blockPath(path.begin(), path.end() - 1);
				auto block = _data.read<T>(blockPath);
				if (block.empty()) {
					throw std::runtime_error("Tensor: empty read result for scalar conversion.");
				}
				if (static_cast<int64_t>(block.size()) <= elementIndex) {
					throw std::runtime_error("Tensor: scalar index out of range.");
				}
				return block[static_cast<size_t>(elementIndex)];
			}
			auto v = _data.read<T>(path);
			if (v.empty()) {
				throw std::runtime_error("Tensor: empty read result for scalar conversion.");
			}
			return v[0];
		}

		void move_from(Tensor&& other) noexcept {
			_meta = std::move(other._meta);
			_data = std::move(other._data);
		}
	};
}