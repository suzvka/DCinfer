#include <stdexcept>
#include <numeric>
#include "TensorMods.h"

namespace DC {

    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // TensorMeta implementation
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    TensorMeta::TensorMeta() {
        ensureTypeMap();
    }

    void TensorMeta::ensureTypeMap() {
        static std::once_flag flag;
        std::call_once(flag, []() { setTypeMap(); });
    }

    bool TensorMeta::check(const std::vector<int64_t>& currentShape) const {
        // Unset rule: skip check
        if (shape.empty()) {
            return true;
        }
        // Dimension count must match
        if (shape.size() != currentShape.size()) {
            return false;
        }
        for (size_t i = 0; i < shape.size(); ++i) {
            // -1 means dynamic dimension: skip comparison
            if (shape[i] != -1 && shape[i] != currentShape[i]) {
                return false;
            }
        }
        return true;
    }

    void TensorMeta::setTypeMap() {
        Type::registerType<float>		(TensorType::Float);
        Type::registerType<double>		(TensorType::Float);

        Type::registerType<int64_t>		(TensorType::Int);
        Type::registerType<int32_t>		(TensorType::Int);
        Type::registerType<int16_t>		(TensorType::Int);
        Type::registerType<int8_t>		(TensorType::Int);
        Type::registerType<int>			(TensorType::Int);

        Type::registerType<uint64_t>	(TensorType::Uint);
        Type::registerType<uint32_t>	(TensorType::Uint);
        Type::registerType<uint16_t>	(TensorType::Uint);
        Type::registerType<uint8_t>		(TensorType::Uint);
        Type::registerType<unsigned int>(TensorType::Uint);

        Type::registerType<bool>		(TensorType::Bool);

        Type::registerType<char>		(TensorType::Char);
        Type::registerType<unsigned char>(TensorType::Char);

        Type::registerType<std::vector<std::byte>>(TensorType::Data);
        Type::registerType<std::vector<char>>	(TensorType::Data);
        Type::registerType<std::vector<float>>	(TensorType::Data);
    }

    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // TensorData implementation
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    TensorData::TensorData() :_data({}) {}

    TensorData::TensorData(
        const std::vector<int64_t>& shape,
        size_t typeSize,
        std::vector<char>&& denseBytes
    ) {
        if (shape.empty() || denseBytes.empty()) {
            return;
        }
        if (typeSize == 0) {
            throw std::invalid_argument("TensorData: typeSize must be > 0");
        }
        for (auto d : shape) {
            if (d <= 0) {
                throw std::invalid_argument("TensorData: shape elements must be > 0");
            }
        }

        size_t elementCount = 1;
        for (auto d : shape) {
            elementCount *= static_cast<size_t>(d);
        }
        const size_t expectedBytes = elementCount * typeSize;
        if (denseBytes.size() != expectedBytes) {
            throw std::invalid_argument("TensorData: denseBytes.size() does not match shape product");
        }

        setDenseBytes(shape, typeSize, std::move(denseBytes));
    }

    TensorData::TensorData(
        const std::vector<int64_t>& shape,
        std::vector<char>&& data
    ) {
        if (shape.empty() || data.empty()) {
            return;
        }

        const int64_t lastDim = shape.back();
        if (lastDim <= 0) {
            throw std::invalid_argument("TensorData: shape.back() must be > 0");
        }
        const size_t blockSize = static_cast<size_t>(lastDim);
        if (data.size() % blockSize != 0) {
            throw std::invalid_argument("TensorData: data.size() must be divisible by shape.back() (block element count)");
        }
        _typeSize = data.size() / blockSize;

        // If shape has only one dimension, the whole data is one block
        if (shape.size() == 1) {
            _isScalar = false;
            _dataSize = data.size();
			// 1D convention: the only block path is {} (rank-1 == 0)
			_data[{}] = std::move(data);
			_dataDimSets.clear();
            _size = _dataSize;
            return;
        }

        // The last dimension is the block size (element count)
        const size_t blockElementCount = blockSize;
        _dataSize = blockElementCount * _typeSize;

        // Path dimensions
        std::vector<int64_t> pathShape(shape.begin(), shape.end() - 1);
        size_t pathDimCount = pathShape.size();
        _dataDimSets.resize(pathDimCount);

        std::vector<int64_t> currentPath(pathDimCount, 0);
        size_t dataOffset = 0;

        const size_t expectedTotalSize = _dataSize * static_cast<size_t>(std::accumulate(pathShape.begin(), pathShape.end(), int64_t{1},
            [](int64_t a, int64_t b) { return a * b; }));
        if (expectedTotalSize != data.size()) {
            throw std::invalid_argument("TensorData: data.size() does not match shape product");
        }

        while (true) {
            // Register current path
            for (size_t i = 0; i < pathDimCount; ++i) {
                _dataDimSets[i].insert(currentPath[i]);
            }

            // Extract and store data block
            DataBlock block(data.begin() + dataOffset, data.begin() + dataOffset + _dataSize);
            _data[currentPath] = std::move(block);
            _size += _dataSize;
            dataOffset += _dataSize;

            // Update to next path (simulate carry)
            int64_t currentDim = static_cast<int64_t>(pathDimCount) - 1;
            while (currentDim >= 0) {
                currentPath[currentDim]++;
                if (currentPath[currentDim] < pathShape[currentDim]) {
                    break; // No carry needed, found next path
                }
                currentPath[currentDim] = 0; // Reset current dimension to 0 and carry to higher dimension
                --currentDim;
            }

            // If all dimensions traversed (carry overflowed highest dimension), break
            if (currentDim < 0) {
                break;
            }
        }
    }

    bool TensorData::cacheValid() const {
        return _cacheValid;
    }

    std::span<const char> TensorData::dataSpan() const {
        if (!_cacheValid) {
            return {};
        }
        return std::span<const char>(_dataCache.data(), _dataCache.size());
    }

    std::span<char> TensorData::dataSpanMut() {
        CacheScope scope(*this, CacheScope::Mode::DenseEdit);
        return std::span<char>(_dataCache.data(), _dataCache.size());
    }

    size_t TensorData::size() const {
        return _size;
    }

    TensorData::DataDimSets TensorData::getDataDimSets() const {
        return _dataDimSets;
    }

    void TensorData::setTypeSize(size_t typeSize) {
        _typeSize = typeSize;
    }

    bool TensorData::writeBytes(std::vector<int64_t> path, size_t ruleTypeSize, const std::vector<char>& bytes) {
        CacheScope scope(*this, CacheScope::Mode::SparseEdit);
        if (ruleTypeSize == 0) {
            throw std::invalid_argument("TensorData::writeBytes: ruleTypeSize must be > 0");
        }
        if (_typeSize == 0) {
            throw std::runtime_error("TensorData::writeBytes: typeSize must be determined at construction");
        }
        if (ruleTypeSize != _typeSize) {
            throw std::invalid_argument("TensorData::writeBytes: ruleTypeSize mismatch");
        }
        if (path.size() != _dataDimSets.size()) {
            clear();
            _dataDimSets.resize(path.size());
        }

        for (int index = 0; index < static_cast<int>(path.size()); ++index) {

            const auto p = path[static_cast<size_t>(index)];
            if (p < 0) return false;
            _dataDimSets[static_cast<size_t>(index)].insert(static_cast<uint64_t>(p));
        }

		{
			const auto it = _data.find(path);
			if (it != _data.end()) {
				_size -= it->second.size();
			}
			_data[path] = bytes;
			_size += _data[path].size();
			if (_dataSize < _data[path].size()) {
				_dataSize = _data[path].size();
			}
		}
        return true;
    }

    void TensorData::clear() {
        _data.clear();
        _dataDimSets.clear();
        _size = 0;
        _dataSize = 0;
        invalidateDenseCache();
        _sparseValid = true;
    }

    std::vector<int64_t> TensorData::getCurrentShape() const {
        if (_cacheValid) {
            return _shapeCache;
        }

		if (_isScalar) {
			return {};
		}

        std::vector<int64_t> shape(_dataDimSets.size(), 0);
        for (size_t i = 0; i < _dataDimSets.size(); ++i) {
            if (!_dataDimSets[i].empty()) {
                shape[i] = static_cast<int64_t>(*std::max_element(_dataDimSets[i].begin(), _dataDimSets[i].end())) + 1;
            }
        }
        if (_dataSize > 0 && _typeSize > 0) {
            shape.push_back(static_cast<int64_t>(_dataSize / _typeSize)); // Last dimension is block element count
        }
        return shape;
    }

    void TensorData::setDenseBytes(const std::vector<int64_t>& shape, size_t typeSize, std::vector<char>&& bytes) {
        clear();
        _typeSize = typeSize;
        _dataCache = std::move(bytes);
        _shapeCache = shape;
        _cacheValid = true;
        _sparseValid = false;
        _isScalar = shape.empty();

        // Note: this is "Dense Pass-Through Mode".
        // TODO: Future optimization: materialize _data (block map) lazily from _flattenedCache + shape only when sparse edit/path query is really needed.
        
        // Sync basic stats
        _dataSize = 0;
        if (!shape.empty() && _typeSize > 0) {
            _dataSize = shape.back() * _typeSize;
        }
        _size = _dataCache.size();
        _dataDimSets.resize(shape.size() > 0 ? shape.size() - 1 : 0);
        for (size_t i = 0; i + 1 < shape.size(); ++i) {
            for (int64_t idx = 0; idx < shape[i]; ++idx) {
                _dataDimSets[i].insert(static_cast<uint64_t>(idx));
            }
        }
    }

    void TensorData::ensureEditable() {
        ensureSparseEditable();
    }

	bool TensorData::isScalar() const {
		return _isScalar;
	}

	void TensorData::setScalar(bool scalar) {
		_isScalar = scalar;
	}

    bool TensorData::writeScalarDenseBytes(const std::vector<int64_t>& path, size_t ruleTypeSize, const void* scalarBytes, size_t scalarByteCount) {
        CacheScope scope(*this, CacheScope::Mode::DenseEdit, CacheScope::DensePolicy::RequireCache);
        if (!scope.active()) {
            return false;
        }
        if (ruleTypeSize == 0) {
            throw std::invalid_argument("TensorData::writeScalarDenseBytes: ruleTypeSize must be > 0");
        }
		if (_typeSize == 0) {
			throw std::runtime_error("TensorData::writeScalarDenseBytes: typeSize must be determined at construction");
		}
		if (ruleTypeSize != _typeSize) {
			throw std::invalid_argument("TensorData::writeScalarDenseBytes: ruleTypeSize mismatch");
		}
        if (_shapeCache.empty()) {
            return false;
        }
        if (path.size() != _shapeCache.size()) {
            return false;
        }
        for (size_t i = 0; i < path.size(); ++i) {
            const auto idx = path[i];
            if (idx < 0 || idx >= _shapeCache[i]) {
                return false;
            }
        }

        const auto& denseShape = _shapeCache;
        const size_t offsetBytes = calculateElementOffsetBytes(path, denseShape);
        if (offsetBytes + _typeSize > _dataCache.size()) {
            return false;
        }

        std::vector<char> tmp(_typeSize, 0);
        std::memcpy(tmp.data(), scalarBytes, std::min(scalarByteCount, _typeSize));
        std::memcpy(_dataCache.data() + offsetBytes, tmp.data(), _typeSize);
        return true;
    }

    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // CacheScope implementation
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    TensorData::CacheScope::CacheScope(TensorData& owner, Mode mode, DensePolicy policy)
        : _owner(&owner), _mode(mode) {
        if (_mode == Mode::DenseEdit) {
            if (policy == DensePolicy::RequireCache && !_owner->_cacheValid) {
                _active = false;
                return;
            }
            _owner->ensureDenseCache();
            _owner->_sparseValid = false;
        }
        else {
            _owner->ensureSparseEditable();
            _owner->_sparseValid = true;
        }
    }

    TensorData::CacheScope::~CacheScope() {
        if (!_active) {
            return;
        }
        if (_mode == Mode::SparseEdit) {
            _owner->invalidateDenseCache();
        }
    }

    bool TensorData::CacheScope::active() const {
        return _active;
    }

    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // Private helpers implementation
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    void TensorData::syncDenseCacheMeta(const std::vector<int64_t>& denseShape) {
        _shapeCache = denseShape;
        _dataSize = 0;
        if (!denseShape.empty() && _typeSize > 0) {
            _dataSize = static_cast<size_t>(denseShape.back()) * _typeSize;
        }
        _size = _dataCache.size();
    }

    void TensorData::invalidateDenseCache() {
        _cacheValid = false;
        _dataCache.clear();
        _shapeCache.clear();
    }

    void TensorData::ensureDenseCache() {
        if (!_cacheValid) {
            buildFlattenedCache();
            _cacheValid = true;
        }
    }

    void TensorData::ensureSparseEditable() {
        if (_cacheValid) {
            materializeFromDense();
            _cacheValid = false;
            _sparseValid = true;
            return;
        }
        if (!_sparseValid) {
            throw std::runtime_error("TensorData: sparse state is invalid without dense cache.");
        }
    }

    size_t TensorData::calculateTotalSize() const {
        size_t totalSize = 1;
        for (const auto& dimSet : _dataDimSets) {
            totalSize *= dimSet.size();
        }
        return totalSize * _dataSize;
    }

    size_t TensorData::calculateBlockOffsetBytes(const std::vector<int64_t>& blockPath, const std::vector<int64_t>& denseShape) const {
        if (denseShape.empty()) {
            throw std::invalid_argument("TensorData::calculateBlockOffsetBytes: denseShape is empty");
        }
        if (denseShape.size() == 1) {
            if (!blockPath.empty()) {
                throw std::out_of_range("TensorData::calculateBlockOffsetBytes: blockPath rank mismatch");
            }
            return 0;
        }
        if (blockPath.size() + 1 != denseShape.size()) {
            throw std::out_of_range("TensorData::calculateBlockOffsetBytes: blockPath rank mismatch");
        }

        size_t elementOffset = 0;
        size_t multiplier = static_cast<size_t>(denseShape.back());
        for (size_t k = blockPath.size(); k-- > 0;) {
            elementOffset += static_cast<size_t>(blockPath[k]) * multiplier;
            multiplier *= static_cast<size_t>(denseShape[k]);
        }
        return elementOffset * _typeSize;
    }

    size_t TensorData::calculateElementOffsetBytes(const std::vector<int64_t>& elementPath, const std::vector<int64_t>& denseShape) const {
        if (denseShape.empty()) {
            if (!elementPath.empty()) {
                throw std::out_of_range("TensorData::calculateElementOffsetBytes: elementPath rank mismatch");
            }
            return 0;
        }
        if (elementPath.size() != denseShape.size()) {
            throw std::out_of_range("TensorData::calculateElementOffsetBytes: elementPath rank mismatch");
        }
        if (denseShape.size() == 1) {
            return static_cast<size_t>(elementPath[0]) * _typeSize;
        }
        const std::vector<int64_t> blockPath(elementPath.begin(), elementPath.end() - 1);
        const size_t blockBase = calculateBlockOffsetBytes(blockPath, denseShape);
        return blockBase + static_cast<size_t>(elementPath.back()) * _typeSize;
    }

    std::vector<int64_t> TensorData::getDenseShape() const {
		if (_isScalar) {
			return {};
		}
        std::vector<int64_t> shape(_dataDimSets.size());
        for (size_t i = 0; i < _dataDimSets.size(); ++i) {
            if (_dataDimSets[i].empty()) {
                shape[i] = 0; // if dimension has no index, size is 0
            }
            else {
                // Shape size is max index + 1
                shape[i] = *std::max_element(_dataDimSets[i].begin(), _dataDimSets[i].end()) + 1;
            }
        }
        if (_dataSize > 0) {
            shape.push_back(_dataSize / _typeSize);
        }
        return shape;
    }

    void TensorData::buildFlattenedCache() {
        const auto denseShape = getDenseShape();

        if (denseShape.empty() || _typeSize == 0) {
            _dataCache.clear();
            syncDenseCacheMeta({});
            return;
        }

        size_t totalElements = 1;
        for (int64_t dimSize : denseShape) {
            if (dimSize <= 0) {
                totalElements = 0;
                break;
            }
            totalElements *= static_cast<size_t>(dimSize);
        }
        const size_t totalBytes = totalElements * _typeSize;

        _dataCache.assign(totalBytes, 0);
        for (const auto& [path, block] : _data) {
            const size_t offset = calculateBlockOffsetBytes(path, denseShape);
			const size_t copyBytes = std::min(block.size(), _dataSize);
			if (offset + copyBytes <= _dataCache.size()) {
				std::memcpy(_dataCache.data() + offset, block.data(), copyBytes);
			}
        }
        syncDenseCacheMeta(denseShape);
    }

    void TensorData::materializeFromDense() {
        if (!_cacheValid) {
            return;
        }
        if (_shapeCache.empty()) {
            return;
        }
        _sparseValid = true;
        _data.clear();
        _dataDimSets.clear();
        _size = 0;

        // Block size: last dim count * typeSize
        _dataSize = static_cast<size_t>(_shapeCache.back()) * _typeSize;
        if (_dataSize == 0) {
            return;
        }

        const size_t pathDims = (_shapeCache.size() >= 2) ? (_shapeCache.size() - 1) : 0;
        _dataDimSets.resize(pathDims);
        for (size_t i = 0; i < pathDims; ++i) {
            for (int64_t idx = 0; idx < _shapeCache[i]; ++idx) {
                _dataDimSets[i].insert(static_cast<uint64_t>(idx));
            }
        }

        if (pathDims == 0) {
            // Degrade to 1D: store whole block at root path {}
            _data[{}] = _dataCache;
            _size = _dataCache.size();
            return;
        }

        std::vector<int64_t> path(pathDims, 0);
        size_t offsetBytes = 0;
        while (true) {
            if (offsetBytes + _dataSize > _dataCache.size()) {
                break;
            }
            DataBlock block(_dataCache.begin() + offsetBytes, _dataCache.begin() + offsetBytes + _dataSize);
            _data[path] = std::move(block);
            _size += _dataSize;
            offsetBytes += _dataSize;

            int64_t dim = static_cast<int64_t>(pathDims) - 1;
            while (dim >= 0) {
                path[static_cast<size_t>(dim)]++;
                // Use shape bounds, not data bytes.
                if (path[static_cast<size_t>(dim)] < _shapeCache[static_cast<size_t>(dim)]) {
                    break;
                }
                path[static_cast<size_t>(dim)] = 0;
                --dim;
            }
            if (dim < 0) {
                break;
            }
        }
    }

}