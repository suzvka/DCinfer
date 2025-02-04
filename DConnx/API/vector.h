#pragma once
#include <vector>
#include <deque>
#include <map>
#include <unordered_set>
#include <stdexcept>
#include <numeric>

namespace DC {
	// 张量数据库
	// 储存实际数据
	// 这是主要对象
	// 需要指定类型
	// 进行编译时类型检查
	template<typename T>
	class tensorPro {
		template<typename L>
		class tensorDim;
	public:
		tensorPro() :_shape({}), _data({}) {}
		// 直接设置致密数据
		// - 形状
		// - 一维数据
		void set(
			const std::vector<uint64_t>& shape, 
			const std::vector<T>& data
		) {
			// 清空原有状态
			_data.clear();
			_shape.clear();
			_record.clear();
			_size = 0;
			// 基础验证
			if (shape.empty()) throw std::invalid_argument("Shape cannot be empty");
			if (shape.back() == 0) throw std::invalid_argument("Block size cannot be zero");
			// 计算结构维度
			const size_t num_structure_dims = shape.size() > 1 ? shape.size() - 1 : 0;
			const uint64_t block_size = shape.back();
			// 直接静态设置形状
			if (num_structure_dims > 0) {
				_shape.assign(shape.begin(), shape.begin() + num_structure_dims);
				// 预填充完整索引记录
				_record.resize(num_structure_dims);
				for (size_t dim = 0; dim < num_structure_dims; ++dim) {
					const uint64_t max_idx = shape[dim];
					for (uint64_t idx = 1; idx <= max_idx; ++idx) {
						_record[dim].insert(idx);
					}
				}
			}
			// 计算总数据块数
			const uint64_t total_blocks = std::accumulate(
				shape.begin(), shape.begin() + num_structure_dims,
				1ULL, std::multiplies<uint64_t>()
			);
			//// 数据量验证
			//if (data.size() != total_blocks * block_size) {
			//	throw std::invalid_argument(
			//		"Data size mismatch. Expected: " +
			//		std::to_string(total_blocks * block_size) +   
			//		", Got: " + std::to_string(data.size())
			//	);
			//}
			// 生成所有路径
			std::vector<std::deque<uint64_t>> all_paths;
			if (num_structure_dims > 0) {
				all_paths = generate_all_paths(_shape);
			}
			else {
				all_paths.push_back({});  // 标量路径
			}
			// 直接填充数据块
			auto data_it = data.begin();
			for (const auto& path : all_paths) {
				const auto end_it = data_it + block_size;
				_data[path] = std::vector<T>(data_it, end_it);
				data_it = end_it;
			}
			_size = block_size;
		}
		// 索引维度
		tensorDim<T>& operator[](uint64_t index) {
			auto i =  new tensorDim<T>(index, this);
			return *i;
        }
		// 获取当前实际形状
		std::vector<uint64_t> getShape() const {
			if (_shape.empty()) {
				return _shape;
			}
			std::vector<uint64_t>temp;
			for (auto i: _record) {
				temp.push_back(i.size());
			}
			temp.push_back(_size);
			return temp;
		}
		// 获取一维数据
		// 按位强转
		// 行优先
		template<typename Y>
		const std::vector<Y> getData() const {
			std::vector<Y> result;
			// 计算张量的元素总数
			size_t totalElements = 1;
			for (auto dim : _shape) {
				totalElements *= dim;
			}
			// 按照行优先遍历所有维度
			std::vector<uint64_t> indices(_shape.size(), 1);  // 用于存储当前维度的索引
			for (size_t i = 0; i < totalElements; ++i) {
				// 获取当前索引对应的数据
				const std::vector<T>* data = getDataAtIndices(indices);
				if (data) {
					for (const T& element : *data) {
						// 先转成字节流
						std::vector<uint8_t> byteStream;
						byteStream.insert(
							byteStream.end(), 
							reinterpret_cast<const uint8_t*>(&element),
							reinterpret_cast<const uint8_t*>(&element) + sizeof(T)
						);
						// 再转成目标类型
						size_t totalSize = byteStream.size();
						for (size_t i = 0; i < totalSize; i += sizeof(Y)) {
							Y* targetElement = reinterpret_cast<Y*>(&byteStream[i]);
							result.push_back(*targetElement);
						}
					}
				}
				// 增加索引
				incrementIndices(indices);
			}
			return result;
		}
		// 默认返回字节形式记录的一维数据
		const std::vector<char> getData() const {
			return getData<char>();
		}
		// 返回指定数据块中的数据
		const std::vector<T>& getData(std::deque<uint64_t>& shape) const {
			if (_data.find(shape) != _data.end()) {
				return _data[shape];
			}
			return nullptr;
		}
	private:
		std::vector<uint64_t> _shape;		// 形状容器
		std::vector<std::unordered_set<uint64_t>> _record; // 实际维度记录器
		// Todo：改成 unordered_map，需要自己写哈希方法
		std::map<
			std::deque<uint64_t>,			// 路径
			std::vector<T>					// 数据块
		> _data;	// 路径 - 数据块
		uint64_t _size = 0;					// 数据块大小

		// 生成所有可能的路径组合
		std::vector<std::deque<uint64_t>> generate_all_paths(const std::vector<uint64_t>& dim_sizes) {
			std::vector<std::deque<uint64_t>> paths;
			std::deque<uint64_t> current_path;
			generate_paths_recursive(dim_sizes, 0, current_path, paths);
			return paths;
		}
		// 递归生成
		void generate_paths_recursive(
			const std::vector<uint64_t>& dim_sizes,
			size_t current_dim,
			std::deque<uint64_t>& current_path,
			std::vector<std::deque<uint64_t>>& result
		) {
			if (current_dim == dim_sizes.size()) {
				result.push_back(current_path);
				return;
			}

			for (uint64_t idx = 1; idx <= dim_sizes[current_dim]; ++idx) {
				current_path.push_back(idx);
				generate_paths_recursive(dim_sizes, current_dim + 1, current_path, result);
				current_path.pop_back();
			}
		}

		// 修改张量数据
		void up(const T& data, std::deque<uint64_t> shape = {}) {
			if (shape.size() != _shape.size()) {
				_size = 0;
				_data.clear();
			}
			updateShape(shape);
			_data[shape] = { data };
		}
		void up(const std::vector<T>& data, std::deque<uint64_t> shape = {}) {
			if (
				shape.size() != _shape.size()||
				data.size()	 != _size
			) {
				_size = 0;
				_data.clear();
			}
			_size = data.size();
			updateShape(shape);
			_data[shape] = data;
		}

		// 根据访问路径更新形状
		void updateShape(std::deque<uint64_t> shape = {}) {
			if (shape.size() != _shape.size()) {
				_shape.clear();
			}
			int i = 0;
			for (auto dimNum : shape) {
				// 标量没有形状
				if (dimNum == 0) {
					_shape.clear();
					_record.clear();
					return;
				}
				// 访问的维度有效时
				if (_shape.size() >= i + 1) {
					dimNum > _shape[i] ? _shape[i] = dimNum : i; // 更新该维度中的最大元素数量
				}
				else {
					// 无效时维度 +1
					_shape.push_back(dimNum);
				}
				i++;
			}
			recordDim(shape);
		}

		// 获取给定维度索引的数据
		const std::vector<T>* getDataAtIndices(const std::vector<uint64_t>& indices) const {
			std::deque<uint64_t> shape(indices.begin(), indices.end());
			if (shape.empty()) {
				shape.push_back(0);
			}
			auto it = _data.find(shape); // 使用find获取迭代器
			if (it != _data.end()) {
				return &(it->second); // 通过迭代器访问元素
			}
			return nullptr;
		}

		// 增加维度索引，进行行优先的遍历
		void incrementIndices(std::vector<uint64_t>& indices) const {
			for (int i = indices.size() - 1; i >= 0; --i) {
				if (indices[i] < _shape[i]) {
					++indices[i];
					return;
				}
				indices[i] = 1;
			}
		}
		// 记录实际形状
		void recordDim(std::deque<uint64_t> shape) {
			while (shape.size() - _record.size() > 0) {
				_record.push_back({});// 对齐维度
			}
			int i = 0;
			for (auto dimNum : shape) {
				_record[i].insert(dimNum);
				i++;
			}
		}
//==============================================================================
		// 临时维度对象
		// 根据调用参数生成一个查询路径
		// 不储存数据
		// 在每次读写张量时生成，完毕之后销毁
		// 使用普通指针是必要的，因为必须将所有 tensorDim 对象视作一个整体
		// 这样才能保证引用链条销毁时不会因为节点断裂而留下残余
		protected:
		template<typename L>
		class tensorDim {
		public:
		//===============================================================
			// 可以被同类对象或主对象创建
			// 此时获取对应的指针
			tensorDim(int64_t index, tensorPro<L>* topObj ,tensorDim<L>* parentObj = nullptr)
			:ID(index), top(topObj), parent(parentObj) {}
		//===============================================================
			// 继续创建子对象
			tensorDim<L>& operator[](uint64_t index) {
				if (index == 0) {
					throw std::runtime_error("标量没有维度");
				}
				auto i = new tensorDim<L>(index, top, this);
				return *i;
			}
		//===============================================================
			// 允许向指定维度读出数据或者写入向量数据块
			// 写入时检查并更新形状
			// 也就是最大的路径
			// 尝试修改嵌套层数时，删除所有数据块
			// 根据 onnx 输入的特点，同个张量不可能兼容不同嵌套层数的形状
			// 因此直接删掉即可

			// 传入单个值
			// 替换索引路径的数据块
			tensorPro<T>& operator=(L newData) {
				if (parent) {
					parent->up(newData, { ID });
				}
				else if (top) {
					top->up(newData, { ID });
				}
				return *top;
			}
			// 传入向量
			// 替换索引路径的数据块
			tensorPro<T>& operator=(std::vector<T> newData) {
				if (parent) {
					parent->up(newData, { ID });
				}
				else if (top) {
					top->up(newData, { ID });
				}
				return *top;
			}
		
		//===============================================================
			// 链式获取数据
			const L& getData(std::deque<uint64_t> shape = {}) {
				if (parent) {
					shape.push_front(ID);
					return parent->getData(shape);
				}
				else if (top) {
					shape.push_front(ID);
					return top->getData(shape);
				}
				return {};
			}
			operator L() { return getData(); }
		//
			tensorPro<T>& Add(L data) {
				*this = data;
				return *top;
			}
			tensorPro<T>& Add(std::vector<T> data) {
				*this = data;
				return *top;
			}
		//===============================================================
		private:
			uint64_t ID = 0;				// 所在位置
			tensorDim<L>* parent = nullptr;	// 父级对象指针
			tensorPro<L>* top = nullptr;;		// 顶级对象指针

			void up(const L& newData, std::deque<uint64_t> shape = {}) {
				shape.push_front(ID);
				if (parent) {
					parent->up(newData, shape);
				}
				else if (top) {
					top->up(newData, shape);
				}
			}
			void up(const std::vector<T>& newData, std::deque<uint64_t> shape = {}) {
				shape.push_front(ID);
				if (parent) {
					parent->up(newData, shape);
				}
				else if (top) {
					top->up(newData, shape);
				}
			}
			//===============================================================
			// 销毁自己
			void del() {
				this->~tensorDim();
			}
		};
	};
//==============================================================================
}