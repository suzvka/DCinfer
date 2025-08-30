#pragma once
#include <vector>
#include <deque>
#include <map>
#include <unordered_set>
#include <string>
#include <memory>
#include <Type.h>

namespace DC {
	// 张量元数据
	struct TensorMeta {
		static enum class TensorType {
			Float,
			Int,
			Uint,
			Bool,
			Char,
			Data,
			Void
		};

		static TypeManager<TensorMeta::TensorType> _typeMap;

		std::string name = "";
		std::string typeName = "";
		size_t typeSize = 0;
		std::vector<int64_t> shape = {};
		TensorType type = TensorType::Void;

		// 检查形状是否符合规则
		bool check(const std::vector<int64_t>& currentShape) const {
			if (shape.empty()) {
				return true;
			}
			if (shape.size() != currentShape.size()) {
				return false;
			}
			for (size_t i = 0; i < shape.size(); ++i) {
				if (shape[i] != -1 && shape[i] != currentShape[i]) {
					return false;
				}
			}
			return true;
		}

		static void setTypeMap() {
			_typeMap.registerType<float>	(TensorType::Float	,"Float");
			_typeMap.registerType<int64_t>	(TensorType::Int	,"Int"	);
			_typeMap.registerType<uint64_t>	(TensorType::Uint	,"UInt"	);
			_typeMap.registerType<bool>		(TensorType::Bool	,"Bool"	);
			_typeMap.registerType<char>		(TensorType::Char	,"Char"	);
			_typeMap.registerType<std::byte>(TensorType::Data	,"Data"	);
			_typeMap.registerType<uint8_t>	(TensorType::Void	,"Void"	);
		}

		TensorMeta() {
			setTypeMap();
		}
	};

	// 用于嵌套引用
	// 根据编辑器中的嵌套层数，自动生成合适的索引形状
	// 顶层张量继承这个类，然后组合 TensorData
	// 之后这里的信息可用于 TensorData 索引张量数据
	class TensorDim {
	public:
		// 作为顶层构造时
		TensorDim() {}

		// 嵌套构造
		TensorDim(TensorDim& parent, uint64_t index)
			: _parent(&parent), _index(index) {
		}

		// 重载引用运算符
		// 但是返回堆上的新 TensorDim
		TensorDim& operator[](uint64_t index) {
			_child = std::make_unique<TensorDim>(*this, index);
			return *_child;
		}

		// 获取形状
		// 逐级上报自己的索引，过程中传递一个路径向量
		std::vector<int64_t> getPath(std::vector<int64_t> path = {}) const {
			path.push_back(_index);
			if (_parent) {
				return _parent->getPath(path);
			}
			else {
				std::reverse(path.begin(), path.end());
				return path;
			}
		}

	private:
		int _index = 0;						// 当前维度索引
		TensorDim* _parent = nullptr;		// 父级维度指针
		std::unique_ptr<TensorDim> _child;	// 子级维度指针

	};

	// 实际储存张量数据的类
	// 数据块一维平摊储存
	class TensorData {
	public:
		using DataDimSets = std::vector<std::unordered_set<uint64_t>>;
		using DataBlock = std::vector<char>;
		using DataMap = std::map<std::vector<int64_t>, DataBlock>;

		TensorData() :_data({}) {}
		TensorData(
			const std::vector<int64_t>& shape,
			std::vector<char>&& data
		) {
			if (shape.empty() || data.empty()) {
				return;
			}

			size_t blockSize = *shape.cend();
			_typeSize = data.size() / blockSize;

			// 如果形状只有一个维度，则整个数据视为一个数据块
			if (shape.size() == 1) {
				_dataSize = data.size();
				_data[{0}] = std::move(data);
				_dataDimSets.resize(1);
				_dataDimSets[0].insert(0);
				_size = _dataSize;
				return;
			}

			// 最后一个维度是数据块的大小（元素数量）
			size_t blockElementCount = shape.back();
			_dataSize = blockElementCount * _typeSize;

			// 路径维度
			std::vector<int64_t> pathShape(shape.begin(), shape.end() - 1);
			size_t pathDimCount = pathShape.size();
			_dataDimSets.resize(pathDimCount);

			std::vector<int64_t> currentPath(pathDimCount, 0);
			size_t dataOffset = 0;

			while (true) {
				// 登记当前路径
				for (size_t i = 0; i < pathDimCount; ++i) {
					_dataDimSets[i].insert(currentPath[i]);
				}

				// 提取并存储数据块
				DataBlock block(data.begin() + dataOffset, data.begin() + dataOffset + _dataSize);
				_data[currentPath] = std::move(block);
				_size += _dataSize;
				dataOffset += _dataSize;

				// 更新到下一个路径（模拟进位）
				int currentDim = pathDimCount - 1;
				while (currentDim >= 0) {
					currentPath[currentDim]++;
					if (currentPath[currentDim] < pathShape[currentDim]) {
						break; // 不需要进位，已找到下一个路径
					}
					currentPath[currentDim] = 0; // 当前维度重置为0，并向更高维度进位
					currentDim--;
				}

				// 如果所有维度都已遍历完（最高位也发生了进位），则退出循环
				if (currentDim < 0) {
					break;
				}
			}
		}

		size_t size() const {
			return _size;
		}

		DataDimSets getDataDimSets() const {
			return _dataDimSets;
		}

		// 设置类型字节数
		// 这会重新解释张量形状
		void setTypeSize(size_t typeSize) {
			_typeSize = typeSize;
		}

		// 写入数据
		// 如果写入索引会改变维度数量，则清空旧数据，当作全新张量写入
		template<typename T>
		bool write(std::vector<int64_t> path, const std::vector<T>& data) {
			if( path.size() != _dataDimSets.size()) clear();

			for (int index = 0; index < path.size();++index) {
				if (path[index] < 0) return false;
				_dataDimSets[index].insert(path[index]);
			}

			_data[path] = toCharVector(data);
			_size += _data[path].size();
			if (_dataSize < _data[path].size()) {
				_dataSize = _data[path].size();
			}

			return true;
		}

		// 删除数据
		bool remove(std::vector<int64_t> path) {
			for (int index = 0; index < path.size(); ++index) {
				if (path[index] < 0) return false;
				_dataDimSets[index].erase(path[index]);
			}
			
			auto it = _data.find(path);
			if (it != _data.end()) {
				_size -= it->second.size();
				_data.erase(it);
			}
			return true;
		}

		// 读取数据
		template<typename T>
		std::vector<T> read(std::vector<int64_t> path) const {
			for (int index = 0; index < path.size(); ++index) {
				if (path[index] < 0) return {};
			}

			auto it = _data.find(path);
			if (it != _data.end()) {
				const DataBlock& block = it->second;
				return fromCharVector(block);
			}

			return {};
		}

		// 清空数据
		void clear() {
			_data.clear();
			_dataDimSets.clear();
			_size = 0;
			_dataSize = 0;
		}

		// 获取当前动态形状
		// 即最大的路径 + 数据块大小
		std::vector<int64_t> getCurrentShape() const {
			std::vector<int64_t> shape(_dataDimSets.size(), 0);
			for (size_t i = 0; i < _dataDimSets.size(); ++i) {
				if (!_dataDimSets[i].empty()) {
					shape[i] = *std::max_element(_dataDimSets[i].begin(), _dataDimSets[i].end());
				}
			}
			if (_dataSize > 0) {
				shape.push_back(_dataSize / _typeSize); // 最后一维为数据块大小
			}
			return shape;
		}

		// 获取一维化张量数据
		// 先用 _dataDimSets 计算出一维化后总大小，全部写 0
		// 然后用 _data 写入数据
		// 这样一来稀疏的张量将填充 0
		// 较小的数据块将对齐到最大数据块大小
		template<typename T>
		std::vector<T> getData() const {
			std::vector<int64_t> denseShape = getDenseShape();
			size_t totalSize = 1;
			for (int64_t dim_size : denseShape) {
				totalSize *= dim_size;
			}
			totalSize *= _typeSize;

			std::vector<char> flattenedData(totalSize, 0);
			for (const auto& [path, block] : _data) {
				size_t offset = calculateOffset(path, denseShape);
				if (offset + block.size() <= flattenedData.size()) {
					std::memcpy(flattenedData.data() + offset, block.data(), block.size());
				}
			}
			return fromCharVector<T>(flattenedData);
		}

	private:
		DataMap _data;
		DataDimSets _dataDimSets;
		size_t _size = 0;
		size_t _typeSize = 0; // 类型字节数
		size_t _dataSize = 0; // 单个数据块的大小

		// 强制类型转换
		// 将输入的数据块转换为 char 类型
		template<typename T>
		DataBlock toCharVector(const std::vector<T>& data) {
			DataBlock charData(data.size() * sizeof(T));
			std::memcpy(charData.data(), data.data(), charData.size());
			return charData;
		}

		template<>
		DataBlock toCharVector<bool>(const std::vector<bool>& data) {
			DataBlock charData;
			charData.reserve(data.size());
			for (bool b : data) {
				charData.push_back(static_cast<char>(b));
			}
			return charData;
		}

		// 从数据块中恢复指定类型的数据
		template<typename T>
		std::vector<T> fromCharVector(const DataBlock& block) const {
			std::vector<T> data(block.size() / sizeof(T));
			std::memcpy(data.data(), block.data(), block.size());
			return data;
		}

		template<>
		std::vector<bool> fromCharVector<bool>(const DataBlock& block) const {
			std::vector<bool> data;
			data.reserve(block.size());
			for (char c : block) {
				data.push_back(c != 0);
			}
			return data;
		}

		// 考虑稀疏张量，计算总大小
		// 取 DataDimSets 中每个维度的最大值，与数据块大小相乘
		size_t calculateTotalSize() const {
			size_t totalSize = 1;
			for (const auto& dimSet : _dataDimSets) {
				totalSize *= dimSet.size();
			}
			return totalSize * _dataSize;
		}

		// 传入一个路径向量(不含最后一级的数据块大小)，计算该路径在一维化张量数据中的起始点
		// 这些路径将从 DataDimSets 中获取，数据块大小将从 _dataSize 获取
		size_t calculateOffset(const std::vector<int64_t>& path, const std::vector<int64_t>& denseShape) const {
			size_t offset = 0;
			size_t multiplier = 1;

			// 最后一个维度是数据块内部，所以我们从右往左计算
			// 先计算数据块的步长
			if (denseShape.size() > path.size()) {
				multiplier = denseShape.back(); // 块大小
			}

			for (int i = path.size() - 1; i >= 0; --i) {
				offset += path[i] * multiplier;
				multiplier *= denseShape[i];
			}

			// 最终偏移量要乘以类型字节数
			return offset * _typeSize;
		}

		// 获取稠密形状
		std::vector<int64_t> getDenseShape() const {
			std::vector<int64_t> shape(_dataDimSets.size());
			for (size_t i = 0; i < _dataDimSets.size(); ++i) {
				if (_dataDimSets[i].empty()) {
					shape[i] = 0; // 如果该维度没有索引，则大小为0
				}
				else {
					// 形状大小应为最大索引值 + 1
					shape[i] = *std::max_element(_dataDimSets[i].begin(), _dataDimSets[i].end()) + 1;
				}
			}
			if (_dataSize > 0) {
				shape.push_back(_dataSize / _typeSize);
			}
			return shape;
		}
	};
}