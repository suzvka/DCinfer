# DConnx



%% Mermaid: Tensor \& TensorSlot 功能概览

flowchart LR

&nbsp; subgraph TensorModule\["张量 (核心行为)"]

&nbsp;   CT\["创建张量"] -->|"初始化元数据和数据存储"| TD\["分配数据存储"]

&nbsp;   CT --> SetName\["设置名称"]

&nbsp;   TD --> ShapeFunc\["计算数据位置"]

&nbsp;   LoadData\["加载数据"] -->|"计算写入位置"| ShapeFunc

&nbsp;   LoadData --> TD

&nbsp;   ViewOp\["视图操作"] -->|"生成路径"| Path\["路径"]

&nbsp;   Path --> Write\["写入数据"]

&nbsp;   Path --> WriteScalar\["写入单个值"]

&nbsp;   Write --> EditMode\["进入编辑模式"]

&nbsp;   WriteScalar --> EditMode

&nbsp;   EditMode -->|"计算写入位置"| ShapeFunc

&nbsp;   EditMode --> TensorDataWrite\["写入数据存储"]

&nbsp;   Read\["读取数据"] -->|"计算读取位置"| ShapeFunc

&nbsp;   Read --> TensorDataRead\["读取数据存储"]

&nbsp;   ReadScalar\["读取单个值"] --> CheckType\["验证类型和路径"]

&nbsp;   CheckType -->|"如果路径完整"| ReadElement\["读取元素"]

&nbsp;   CheckType -->|"如果剩余维度为1"| BuildFull\["补全路径"]

&nbsp;   BuildFull --> indexShapeRead\["计算数据位置(读)"]

&nbsp;   indexShapeRead --> ReadElement

&nbsp;   ReadElement --> ReturnVal\["返回数据"]

&nbsp;   IndexShapeNotes\["核心：索引计算"]:::note

&nbsp; end



&nbsp; subgraph SlotModule\["数据槽 (数据通道)"]

&nbsp;   CreateSlot\["创建数据槽"] --> SetRule\["设置名称和形状规则"]

&nbsp;   CreateSlot --> SetType\["设置数据类型"]

&nbsp;   SetDefault\["设置默认张量"] -->|"验证匹配"| StoreDefault\["存储默认张量"]

&nbsp;   InputOp\["输入张量"] --> MatchCheck\["检查类型匹配"]

&nbsp;   MatchCheck -->|"匹配成功"| StoreData\["存储输入数据"]

&nbsp;   StoreData --> HasDataFlag\["数据已存在"]

&nbsp;   GetTensor\["获取张量"] -->|"获取数据"| HasDataCond{"有数据则移出，否则返回默认"}

&nbsp;   HasDataCond --> MoveOut\["移出数据"]

&nbsp;   HasDataCond --> ReturnDefault\["返回默认数据"]

&nbsp;   ClearOps\["清除数据"] --> ResetPtrs\["重置指针"]

&nbsp;   OutputOp\["输出张量"] --> GetTensor

&nbsp; end



&nbsp; %% 错误处理

&nbsp; CT -.->|"遇到错误"| Abort\["抛出异常"]

&nbsp; ReadScalar -.->|"类型错误/路径无效/不是标量"| Abort

&nbsp; MatchCheck -.->|"类型或形状不匹配"| AbortSlot\["抛出异常"]



&nbsp; classDef note fill:#f9f,stroke:#333,stroke-width:1px

&nbsp; class IndexShapeNotes note

