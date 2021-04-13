/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { onUnexpectedExternalError, canceled, isPromiseCanceledError } from 'vs/base/common/errors';
import { IEditorContribution } from 'vs/editor/common/editorCommon';
import { ITextModel } from 'vs/editor/common/model';
import * as modes from 'vs/editor/common/modes';
import { Position, IPosition } from 'vs/editor/common/core/position';
import { RawContextKey } from 'vs/platform/contextkey/common/contextkey';
import { ICodeEditor } from 'vs/editor/browser/editorBrowser';
import { CancellationToken } from 'vs/base/common/cancellation';
import { Range } from 'vs/editor/common/core/range';
import { FuzzyScore } from 'vs/base/common/filters';
import { isDisposable, DisposableStore, IDisposable } from 'vs/base/common/lifecycle';
import { MenuId } from 'vs/platform/actions/common/actions';
import { SnippetParser } from 'vs/editor/contrib/snippet/snippetParser';
import { StopWatch } from 'vs/base/common/stopwatch';
import { CommandsRegistry } from 'vs/platform/commands/common/commands';
import { assertType } from 'vs/base/common/types';
import { URI } from 'vs/base/common/uri';
import { ITextModelService } from 'vs/editor/common/services/resolverService';
import { localize } from 'vs/nls';

export const Context = {
	Visible: new RawContextKey<boolean>('suggestWidgetVisible', false, localize('suggestWidgetVisible', "Whether suggestion are visible")),
	DetailsVisible: new RawContextKey<boolean>('suggestWidgetDetailsVisible', false, localize('suggestWidgetDetailsVisible', "Whether suggestion details are visible")),
	MultipleSuggestions: new RawContextKey<boolean>('suggestWidgetMultipleSuggestions', false, localize('suggestWidgetMultipleSuggestions', "Whether there are multiple suggestions to pick from")),
	MakesTextEdit: new RawContextKey('suggestionMakesTextEdit', true, localize('suggestionMakesTextEdit', "Whether inserting the current suggestion yields in a change or has everything already been typed")),
	AcceptSuggestionsOnEnter: new RawContextKey<boolean>('acceptSuggestionOnEnter', true, localize('acceptSuggestionOnEnter', "Whether suggestions are inserted when pressing Enter")),
	HasInsertAndReplaceRange: new RawContextKey('suggestionHasInsertAndReplaceRange', false, localize('suggestionHasInsertAndReplaceRange', "Whether the current suggestion has insert and replace behaviour")),
	InsertMode: new RawContextKey<'insert' | 'replace'>('suggestionInsertMode', undefined, { type: 'string', description: localize('suggestionInsertMode', "Whether the default behaviour is to insert or replace") }),
	CanResolve: new RawContextKey('suggestionCanResolve', false, localize('suggestionCanResolve', "Whether the current suggestion supports to resolve further details")),
};

export const suggestWidgetStatusbarMenu = new MenuId('suggestWidgetStatusBar');

export class CompletionItem {

	_brand!: 'ISuggestionItem';

	//
	readonly editStart: IPosition;
	readonly editInsertEnd: IPosition;
	readonly editReplaceEnd: IPosition;

	//
	readonly textLabel: string;

	// perf
	readonly labelLow: string;
	readonly sortTextLow?: string;
	readonly filterTextLow?: string;

	// validation
	readonly isInvalid: boolean = false;

	// sorting, filtering
	score: FuzzyScore = FuzzyScore.Default;
	distance: number = 0;
	idx?: number;
	word?: string;

	// resolving
	private _isResolved?: boolean;
	private _resolveCache?: Promise<void>;

	constructor(
		readonly position: IPosition,
		readonly completion: modes.CompletionItem,
		readonly container: modes.CompletionList,
		readonly provider: modes.CompletionItemProvider,
	) {
		this.textLabel = typeof completion.label === 'string'
			? completion.label
			: completion.label.name;

		// ensure lower-variants (perf)
		this.labelLow = this.textLabel.toLowerCase();

		// validate label
		this.isInvalid = !this.textLabel;

		this.sortTextLow = completion.sortText && completion.sortText.toLowerCase();
		this.filterTextLow = completion.filterText && completion.filterText.toLowerCase();

		// normalize ranges
		if (Range.isIRange(completion.range)) {
			this.editStart = new Position(completion.range.startLineNumber, completion.range.startColumn);
			this.editInsertEnd = new Position(completion.range.endLineNumber, completion.range.endColumn);
			this.editReplaceEnd = new Position(completion.range.endLineNumber, completion.range.endColumn);

			// validate range
			this.isInvalid = this.isInvalid
				|| Range.spansMultipleLines(completion.range) || completion.range.startLineNumber !== position.lineNumber;

		} else {
			this.editStart = new Position(completion.range.insert.startLineNumber, completion.range.insert.startColumn);
			this.editInsertEnd = new Position(completion.range.insert.endLineNumber, completion.range.insert.endColumn);
			this.editReplaceEnd = new Position(completion.range.replace.endLineNumber, completion.range.replace.endColumn);

			// validate ranges
			this.isInvalid = this.isInvalid
				|| Range.spansMultipleLines(completion.range.insert) || Range.spansMultipleLines(completion.range.replace)
				|| completion.range.insert.startLineNumber !== position.lineNumber || completion.range.replace.startLineNumber !== position.lineNumber
				|| completion.range.insert.startColumn !== completion.range.replace.startColumn;
		}

		// create the suggestion resolver
		if (typeof provider.resolveCompletionItem !== 'function') {
			this._resolveCache = Promise.resolve();
			this._isResolved = true;
		}
	}

	// ---- resolving

	get isResolved(): boolean {
		return !!this._isResolved;
	}

	async resolve(token: CancellationToken) {
		if (!this._resolveCache) {
			const sub = token.onCancellationRequested(() => {
				this._resolveCache = undefined;
				this._isResolved = false;
			});
			this._resolveCache = Promise.resolve(this.provider.resolveCompletionItem!(this.completion, token)).then(value => {
				Object.assign(this.completion, value);
				this._isResolved = true;
				sub.dispose();
			}, err => {
				if (isPromiseCanceledError(err)) {
					// the IPC queue will reject the request with the
					// cancellation error -> reset cached
					this._resolveCache = undefined;
					this._isResolved = false;
				}
			});
		}
		return this._resolveCache;
	}
}

export const enum SnippetSortOrder {
	Top, Inline, Bottom
}

export class CompletionOptions {

	static readonly default = new CompletionOptions();

	constructor(
		readonly snippetSortOrder = SnippetSortOrder.Bottom,
		readonly kindFilter = new Set<modes.CompletionItemKind>(),
		readonly providerFilter = new Set<modes.CompletionItemProvider>(),
	) { }
}

let _snippetSuggestSupport: modes.CompletionItemProvider;

export function getSnippetSuggestSupport(): modes.CompletionItemProvider {
	return _snippetSuggestSupport;
}

export function setSnippetSuggestSupport(support: modes.CompletionItemProvider): modes.CompletionItemProvider {
	const old = _snippetSuggestSupport;
	_snippetSuggestSupport = support;
	return old;
}

export interface CompletionDurationEntry {
	readonly providerName: string;
	readonly elapsedProvider: number;
	readonly elapsedOverall: number;
}

export interface CompletionDurations {
	readonly entries: readonly CompletionDurationEntry[];
	readonly elapsed: number;
}

export class CompletionItemModel {
	constructor(
		readonly items: CompletionItem[],
		readonly needsClipboard: boolean,
		readonly durations: CompletionDurations,
		readonly disposable: IDisposable,
		readonly inactiveProvider: Set<modes.CompletionItemProvider> | undefined,
	) { }
}

function delay(wait: number) {
	return new Promise((resolve) => setTimeout(resolve, wait));
}
/**
 * update gts 2020-04-12
 * 模拟一个智能提示遮罩
 * @param model
 * @param position
 * @param context
 * @param token
 */
export async function provideSuggestionMaskItems(
	providerSuggestionGroup: ProviderSuggestionGroup,
	model: ITextModel,
	position: Position,
	context: modes.CompletionContext = { triggerKind: modes.CompletionTriggerKind.Invoke },
	token: CancellationToken = CancellationToken.None
) {
	const maskCompletionItemProvider: modes.CompletionItemProvider = {
		provideCompletionItems(model, position, context, token) {
			const wordPos = model.getWordUntilPosition(position);
			const range = {
				startLineNumber: position.lineNumber,
				endLineNumber: position.lineNumber,
				startColumn: wordPos.startColumn,
				endColumn: wordPos.endColumn,
			};
			const suggestions: modes.CompletionItem[] = [
				{
					label: wordPos.word,
					// sortText: 'test',
					// filterText: 'test',
					kind: modes.CompletionItemKind.Function,
					// documentation: '恰当的说明',
					detail: '正在计算中',
					insertText: wordPos.word,
					range,
				},
			];
			// incomplete?: boolean;
			// dispose?(): void;
			// editorIsComposing 后续的多次输入会额外触发 _refilterCompletionItems ，内置会再次屌用 _onNewContext
			// 调用 _onNewContext 中会再次判断是否属于 column 改变 以及 incomplete size 大于 1 ，会触发tigger 进行 column 同步 再次回调 _onNewContext
			return {
				incomplete: true, // 设置为true 意味着下一次上下文 column 的改变
				suggestions,
			};
		}
	};
	const container = maskCompletionItemProvider.provideCompletionItems(model, position, context, token) as modes.CompletionList;
	const completionItemList = providerSuggestionGroup.getCompletionItemList(maskCompletionItemProvider, container);
	return completionItemList.concat(providerSuggestionGroup.completionItemList);
}




/**
 * update gts 2021-04-13
 * 监听智能提示优先检索 主要用于解决尽早的异步请求动作
 */
export type ProviderSuggestionResultGroup = Array<
	Array<{
		provider: modes.CompletionItemProvider,
		list: modes.CompletionList | null | undefined
	} | undefined>
>;

export interface ProviderSuggestionGroup {
	wait: () => Promise<any>;
	completionItemList: Array<CompletionItem>;
	durations: CompletionDurationEntry[];
	disposables: DisposableStore;
	needsClipboard: boolean;
	inactiveProvider: Set<modes.CompletionItemProvider>;
	onCompletionList: (provider: modes.CompletionItemProvider, container: modes.CompletionList | null | undefined, sw: StopWatch) => void;
	getCompletionItemList: (provider: modes.CompletionItemProvider, container: modes.CompletionList | null | undefined) => Array<CompletionItem>;
}
export function providerSuggestionGroupRequest(
	prevItems: CompletionItem[] = [],
	model: ITextModel,
	position: Position,
	options: CompletionOptions = CompletionOptions.default,
	context: modes.CompletionContext = { triggerKind: modes.CompletionTriggerKind.Invoke },
	token: CancellationToken = CancellationToken.None
): ProviderSuggestionGroup {
	position = position.clone();
	const providerResultGroup: ProviderSuggestionResultGroup = [];
	const result: CompletionItem[] = prevItems || [];
	const word = model.getWordAtPosition(position);
	const defaultReplaceRange = word ? new Range(position.lineNumber, word.startColumn, position.lineNumber, word.endColumn) : Range.fromPositions(position);
	const defaultRange = { replace: defaultReplaceRange, insert: defaultReplaceRange.setEndPosition(position.lineNumber, position.column) };
	const disposables = new DisposableStore();
	const durations: CompletionDurationEntry[] = [];
	let needsClipboard = false;

	// 所有的未活动的
	const inactiveProvider = new Set(modes.CompletionProviderRegistry.all(model));

	const getCompletionItemList = (provider: modes.CompletionItemProvider, container: modes.CompletionList | null | undefined) => {
		const completionItems: Array<CompletionItem> = [];
		if (!container) {
			return completionItems;
		}
		for (let suggestion of container.suggestions) {
			if (!options.kindFilter.has(suggestion.kind)) {
				// fill in default range when missing
				if (!suggestion.range) {
					suggestion.range = defaultRange;
				}
				// fill in default sortText when missing
				if (!suggestion.sortText) {
					suggestion.sortText = typeof suggestion.label === 'string' ? suggestion.label : suggestion.label.name;
				}
				if (!needsClipboard && suggestion.insertTextRules && suggestion.insertTextRules & modes.CompletionItemInsertTextRule.InsertAsSnippet) {
					needsClipboard = SnippetParser.guessNeedsClipboard(suggestion.insertText);
				}
				completionItems.push(new CompletionItem(position, suggestion, container, provider));
			}
		}
		return completionItems;
	};

	const onCompletionList = (provider: modes.CompletionItemProvider, container: modes.CompletionList | null | undefined, sw: StopWatch) => {
		if (!container) {
			return;
		}
		const completionItems = getCompletionItemList(provider, container);
		result.push(...completionItems);
		if (isDisposable(container)) {
			disposables.add(container);
		}
		durations.push({
			providerName: provider._debugDisplayName ?? 'unkown_provider', elapsedProvider: container.duration ?? -1, elapsedOverall: sw.elapsed()
		});
	};

	const startTask = async () => {
		// add suggestions from contributed providers - providers are ordered in groups of
		// equal score and once a group produces a result the process stops
		// get provider groups, always add snippet suggestion provider
		// 每次只会获取一次结果
		for (let providerGroup of modes.CompletionProviderRegistry.orderedGroups(model)) {
			// for each support in the group ask for suggestions
			let lenBefore = result.length;

			const providerResults = await Promise.all(providerGroup.map(async provider => {
				if (options.providerFilter.size > 0 && !options.providerFilter.has(provider)) {
					// 不在本次过滤中的 不做为下一次的需要记录活动 进行过滤剔除
					inactiveProvider.delete(provider);
					return;
				}
				try {
					const list = await provider.provideCompletionItems(model, position, context, token);
					const sw = new StopWatch(true);
					onCompletionList(provider, list, sw);
					inactiveProvider.delete(provider);
					return {
						provider,
						list
					};
				} catch (err) {
					onUnexpectedExternalError(err);
				}
				return;
			}));
			if (lenBefore !== result.length || token.isCancellationRequested) {
				break;
			}
			providerResultGroup.push(providerResults);
		}
	};
	const task = startTask();
	/**
	 * 获取当前已经请求完毕的结果集
	 */
	return {
		onCompletionList,
		getCompletionItemList,
		get completionItemList() {
			return result;
		},
		get inactiveProvider() {
			return inactiveProvider;
		},
		get durations() {
			return durations;
		},
		get disposables() {
			return disposables;
		},
		get needsClipboard() {
			return needsClipboard;
		},
		wait() {
			return task;
		}
	};
}

export async function provideSuggestionItems(
	model: ITextModel,
	position: Position,
	options: CompletionOptions = CompletionOptions.default,
	context: modes.CompletionContext = { triggerKind: modes.CompletionTriggerKind.Invoke },
	token: CancellationToken = CancellationToken.None,
	providerSuggestionGroup?: ProviderSuggestionGroup,
): Promise<CompletionItemModel> {

	if (!providerSuggestionGroup) {
		providerSuggestionGroup = providerSuggestionGroupRequest([], model, position, options, context, token);
	}
	const { onCompletionList } = providerSuggestionGroup;

	await delay(0);
	const sw = new StopWatch(true);
	// ask for snippets in parallel to asking "real" providers. Only do something if configured to
	// do so - no snippet filter, no special-providers-only request
	const snippetCompletions = (async () => {
		if (!_snippetSuggestSupport || options.kindFilter.has(modes.CompletionItemKind.Snippet)) {
			return;
		}
		if (options.providerFilter.size > 0 && !options.providerFilter.has(_snippetSuggestSupport)) {
			return;
		}
		const sw = new StopWatch(true);
		const list = await _snippetSuggestSupport.provideCompletionItems(model, position, context, token);
		onCompletionList(_snippetSuggestSupport, list, sw);
	})();

	// 等待所有请求完毕
	await providerSuggestionGroup.wait();
	await snippetCompletions;

	// 取到最终的结果数据
	const { disposables, needsClipboard, durations, completionItemList, inactiveProvider } = providerSuggestionGroup;

	if (token.isCancellationRequested) {
		disposables.dispose();
		return Promise.reject<any>(canceled());
	}
	return new CompletionItemModel(
		completionItemList.sort(getSuggestionComparator(options.snippetSortOrder)),
		needsClipboard,
		{ entries: durations, elapsed: sw.elapsed() },
		disposables,
		inactiveProvider,
	);
}


function defaultComparator(a: CompletionItem, b: CompletionItem): number {
	// check with 'sortText'
	if (a.sortTextLow && b.sortTextLow) {
		if (a.sortTextLow < b.sortTextLow) {
			return -1;
		} else if (a.sortTextLow > b.sortTextLow) {
			return 1;
		}
	}
	// check with 'label'
	if (a.completion.label < b.completion.label) {
		return -1;
	} else if (a.completion.label > b.completion.label) {
		return 1;
	}
	// check with 'type'
	return a.completion.kind - b.completion.kind;
}

function snippetUpComparator(a: CompletionItem, b: CompletionItem): number {
	if (a.completion.kind !== b.completion.kind) {
		if (a.completion.kind === modes.CompletionItemKind.Snippet) {
			return -1;
		} else if (b.completion.kind === modes.CompletionItemKind.Snippet) {
			return 1;
		}
	}
	return defaultComparator(a, b);
}

function snippetDownComparator(a: CompletionItem, b: CompletionItem): number {
	if (a.completion.kind !== b.completion.kind) {
		if (a.completion.kind === modes.CompletionItemKind.Snippet) {
			return 1;
		} else if (b.completion.kind === modes.CompletionItemKind.Snippet) {
			return -1;
		}
	}
	return defaultComparator(a, b);
}

interface Comparator<T> { (a: T, b: T): number; }
const _snippetComparators = new Map<SnippetSortOrder, Comparator<CompletionItem>>();
_snippetComparators.set(SnippetSortOrder.Top, snippetUpComparator);
_snippetComparators.set(SnippetSortOrder.Bottom, snippetDownComparator);
_snippetComparators.set(SnippetSortOrder.Inline, defaultComparator);

export function getSuggestionComparator(snippetConfig: SnippetSortOrder): (a: CompletionItem, b: CompletionItem) => number {
	return _snippetComparators.get(snippetConfig)!;
}

CommandsRegistry.registerCommand('_executeCompletionItemProvider', async (accessor, ...args: [URI, IPosition, string?, number?]) => {
	const [uri, position, triggerCharacter, maxItemsToResolve] = args;
	assertType(URI.isUri(uri));
	assertType(Position.isIPosition(position));
	assertType(typeof triggerCharacter === 'string' || !triggerCharacter);
	assertType(typeof maxItemsToResolve === 'number' || !maxItemsToResolve);

	const ref = await accessor.get(ITextModelService).createModelReference(uri);
	try {

		const result: modes.CompletionList = {
			incomplete: false,
			suggestions: []
		};

		const resolving: Promise<any>[] = [];
		const completions = await provideSuggestionItems(ref.object.textEditorModel, Position.lift(position), undefined, { triggerCharacter, triggerKind: triggerCharacter ? modes.CompletionTriggerKind.TriggerCharacter : modes.CompletionTriggerKind.Invoke });
		for (const item of completions.items) {
			if (resolving.length < (maxItemsToResolve ?? 0)) {
				resolving.push(item.resolve(CancellationToken.None));
			}
			result.incomplete = result.incomplete || item.container.incomplete;
			result.suggestions.push(item.completion);
		}

		try {
			await Promise.all(resolving);
			return result;
		} finally {
			setTimeout(() => completions.disposable.dispose(), 100);
		}

	} finally {
		ref.dispose();
	}

});

interface SuggestController extends IEditorContribution {
	triggerSuggest(onlyFrom?: Set<modes.CompletionItemProvider>): void;
}

const _provider = new class implements modes.CompletionItemProvider {

	onlyOnceSuggestions: modes.CompletionItem[] = [];

	provideCompletionItems(): modes.CompletionList {
		let suggestions = this.onlyOnceSuggestions.slice(0);
		let result = { suggestions };
		this.onlyOnceSuggestions.length = 0;
		return result;
	}
};

modes.CompletionProviderRegistry.register('*', _provider);

export function showSimpleSuggestions(editor: ICodeEditor, suggestions: modes.CompletionItem[]) {
	setTimeout(() => {
		_provider.onlyOnceSuggestions.push(...suggestions);
		editor.getContribution<SuggestController>('editor.contrib.suggestController').triggerSuggest(new Set<modes.CompletionItemProvider>().add(_provider));
	}, 0);
}
