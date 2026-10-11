/**
 * @dcir/graph-editor core 层公共 API（headless）
 *
 * 纪律：本层只含纯函数与类型（无 React / 渲染框架运行时依赖），二次开发可
 * 绕开 ui 层直接消费本层构建自己的界面。扩展一律不修改本层内部：
 * - 数据源经 CatalogProvider 注入（catalog.ts）；
 * - 布局经 LayoutStrategy 注入（layout.ts）；
 * - 校验经 validateEditorValue 产出的 EditorIssue（断 code 不锁文案）。
 */

export * from "./catalog";
export * from "./layout";
export * from "./model";
export * from "./rules";
export * from "./types";
