#pragma once
#include <vector>

namespace DC {
    template <typename T>
    class tree{
    public:
        // 构造时不创建容器，仅保留指针
        tree(){}
        // 如果提供了数据，则直接创建有效节点
        tree(std::vector<T> data){
            if (_dim == nullptr) {
                _dim = new std::vector<tree<T>>();
                _data = new std::vector<T>(data);
            }
        }
        // 析构时回收
        ~tree() {
            //if (_dim != nullptr) delete _dim;
            //if (_data != nullptr) delete _data;
        }
        // 真正用到节点时才创建容器
        tree<T>& operator[](uint64_t index) {
            // 访问下一维度
            if (_dim == nullptr) {
                _dim = new std::vector<tree<T>>();
                // _data = new std::vector<T>();
            }
            while (_dim->size() <= index) {
                _dim->emplace_back(tree<T>());
            }
            return (*_dim)[index];
        }

        T operator() (uint64_t index) {
            // 访问指定维度的数据元素
            if (_dim == nullptr) {
                _dim = new std::vector<tree<T>>();
                _data = new std::vector<T>();
            }
            while (_dim->size() < index) {
                _dim->emplace_back(tree<T>());
            }
            return (*_data)[index];
        }

        tree<T>& operator= (tree<T>& newData) = delete;
        tree<T>& operator= (const std::vector<T>& newData) {
            // 覆盖指定维度的数据
            if (_dim == nullptr) {
                _dim = new std::vector<tree<T>>();
                _data = new std::vector<T>();
            }
            else {
                if (_data != nullptr) delete _data;
            }
            _data = new std::vector<T>(newData);
            return *this;
        }

        const std::vector<T>& get(){
            // 获得指定维度的只读数据向量
            if (_dim == nullptr) {
                _dim = new std::vector<tree<T>>();
                _data = new std::vector<T>();
            }
            return *_data;
        }

    private:
        std::vector<tree<T>>* _dim = nullptr;    // 维度容器
        std::vector<T>* _data = nullptr;         // 数据容器
    };
}