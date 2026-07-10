import { useState, useMemo } from 'react';
import * as Tooltip from '@radix-ui/react-tooltip';
import { registerProvisionDetailRenderer, type ProvisionDetailProps } from './registry';
import type { TicketPriorityDetail } from './octagramTypes';

type FormulaItem = {
  id: string;
  symbol: string;
  value: number;
  title: string;
  description: string;
};

function formatNumber(value?: number): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  if (Number.isInteger(value)) return `${value}`;
  return value.toFixed(2).replace(/\.00$/, '').replace(/(\.\d*[1-9])0+$/, '$1');
}

function parseTicketPriority(data: unknown): TicketPriorityDetail | null {
  if (!data || typeof data !== 'object') return null;
  const record = data as { ticketPriority?: unknown };
  if (!record.ticketPriority || typeof record.ticketPriority !== 'object') return null;

  const ticket = record.ticketPriority as Partial<TicketPriorityDetail>;
  if (typeof ticket.priority !== 'number') return null;
  if (typeof ticket.platformWeight !== 'number') return null;
  if (typeof ticket.bizWeight !== 'number') return null;
  if (typeof ticket.bizPriority !== 'number') return null;
  if (typeof ticket.gpuUtilWeight !== 'number') return null;
  if (typeof ticket.gpuUtilPriority !== 'number') return null;
  if (typeof ticket.resourceGroupWeight !== 'number') return null;
  if (typeof ticket.resourceGroupPriority !== 'number') return null;
  if (typeof ticket.workloadWeight !== 'number') return null;
  if (typeof ticket.workloadPriority !== 'number') return null;
  if (typeof ticket.ticketId !== 'number') return null;

  return ticket as TicketPriorityDetail;
}

// Formula: platformWeight * (bizWeight * (gpuUtilWeight * gpuUtilPriority + resourceGroupWeight * resourceGroupPriority) + workloadWeight * workloadPriority)
function calcPriority(tp: TicketPriorityDetail): number {
  return tp.platformWeight * (
    tp.bizWeight * (
      tp.gpuUtilWeight * tp.gpuUtilPriority +
      tp.resourceGroupWeight * tp.resourceGroupPriority
    ) +
    tp.workloadWeight * tp.workloadPriority
  );
}

function FormulaNumber({ item }: { item: FormulaItem }) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>
        <button
          type="button"
          className="inline underline decoration-dotted underline-offset-[3px] decoration-[#4f5fff] text-[#1d2752] hover:text-[#4f5fff]"
          aria-label={`${item.symbol}: ${item.description}`}
        >
          {formatNumber(item.value)}
        </button>
      </Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content
          sideOffset={6}
          className="z-50 max-w-[360px] rounded-md bg-[#111827] px-3 py-2 text-xs leading-5 text-white shadow-lg"
        >
          <p className="font-medium">{item.title}</p>
          <p className="mt-1 text-gray-200">{item.description}</p>
          <Tooltip.Arrow className="fill-[#111827]" />
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

function parseMatchOrderUrl(data: unknown): string | undefined {
  if (!data || typeof data !== 'object') return undefined;
  const d = data as Record<string, unknown>;
  return typeof d.matchOrderUrl === 'string' ? d.matchOrderUrl : undefined;
}

function OctagramProvisionDetailRenderer({ data }: ProvisionDetailProps) {
  const tp = parseTicketPriority(data);
  const [detailOpen, setDetailOpen] = useState(false);

  const formulaItems = useMemo<FormulaItem[]>(() => {
    if (!tp) return [];
    return [
      {
        id: 'platformWeight',
        symbol: 'Wscene',
        value: tp.platformWeight,
        title: '场景权重',
        description: 'Serving、Training 等不同场景会有不同的权重',
      },
      {
        id: 'bizWeight',
        symbol: 'Wbiz',
        value: tp.bizWeight,
        title: '业务线权重',
        description: '业务线优先级的权重因子',
      },
      {
        id: 'bizPriority',
        symbol: 'Pbiz',
        value: tp.bizPriority,
        title: '业务线优先级',
        description: '撮合平台分配给特定业务线对应的优先级，对同一个业务线，不同的时间，优先级可能不一样',
      },
      {
        id: 'gpuUtilWeight',
        symbol: 'Wgpu',
        value: tp.gpuUtilWeight,
        title: 'GPU 利用率权重',
        description: 'GPU 利用率优先级的权重因子',
      },
      {
        id: 'gpuUtilPriority',
        symbol: 'Pgpu',
        value: tp.gpuUtilPriority,
        title: 'GPU 利用率优先级',
        description: '基于 GPU 利用率分配的优先级',
      },
      {
        id: 'resourceGroupWeight',
        symbol: 'Wrg',
        value: tp.resourceGroupWeight,
        title: '资源组权重',
        description: '资源组优先级的权重因子',
      },
      {
        id: 'resourceGroupPriority',
        symbol: 'Prg',
        value: tp.resourceGroupPriority,
        title: '资源组优先级',
        description: '资源组对应的优先级',
      },
      {
        id: 'workloadWeight',
        symbol: 'Wwl',
        value: tp.workloadWeight,
        title: '工作负载权重',
        description: '工作负载优先级的权重因子',
      },
      {
        id: 'workloadPriority',
        symbol: 'Pwl',
        value: tp.workloadPriority,
        title: '工作负载优先级',
        description: '算力平台分配给业务线内特定工作负载的优先级',
      },
    ];
  }, [tp]);

  const matchOrderUrl = parseMatchOrderUrl(data);

  if (!tp) {
    return null;
  }

  const computed = calcPriority(tp);
  const fusionPriority = typeof tp.priority === 'number' ? tp.priority : computed;
  const staticPriority = tp.bizWeight * tp.bizPriority;
  const utilizationCoeff = tp.gpuUtilWeight * tp.gpuUtilPriority;

  return (
    <div>
      <div className="flex items-end justify-between gap-3">
        <div>
          <p className="text-sm text-gray-500">Fusion Priority</p>
          <p className="mt-1 text-2xl leading-none font-semibold text-[#4f5fff]">{formatNumber(fusionPriority)}</p>
        </div>
        {matchOrderUrl ? (
          <a
            href={matchOrderUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-blue-500 hover:text-blue-600 underline decoration-blue-300 hover:decoration-blue-500 underline-offset-2"
          >
            Matching Order #{formatNumber(tp.ticketId)}
          </a>
        ) : (
          <p className="text-xs text-[#8c92ad]">Matching Order #{formatNumber(tp.ticketId)}</p>
        )}
      </div>

      <Tooltip.Provider delayDuration={180}>
        <div className="mt-4 rounded-md border border-[#eceffd] bg-[#f7f8fd] px-3 py-3 text-[14px] leading-6 text-[#2a3358] flex items-center justify-between">
          <span>
            <span className="font-semibold text-[#4f5fff]">{formatNumber(computed)}</span>
            <span> = </span>
            {formulaItems[0] && <FormulaNumber item={formulaItems[0]} />}
            <span> × (</span>
            {formulaItems[1] && <FormulaNumber item={formulaItems[1]} />}
            <span> × </span>
            {formulaItems[2] && <FormulaNumber item={formulaItems[2]} />}
            <span> + </span>
            {formulaItems[7] && <FormulaNumber item={formulaItems[7]} />}
            <span> × </span>
            {formulaItems[8] && <FormulaNumber item={formulaItems[8]} />}
            <span>)</span>
          </span>
          <button
            type="button"
            onClick={() => setDetailOpen(!detailOpen)}
            className="inline-flex items-center text-gray-400 hover:text-gray-600 shrink-0 ml-2"
          >
            {detailOpen ? (
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-3.5 h-3.5"><path fillRule="evenodd" d="M14.77 12.77a.75.75 0 0 1-1.06 0L10 9.06l-3.71 3.71a.75.75 0 0 1-1.06-1.06l4.24-4.24a.75.75 0 0 1 1.06 0l4.24 4.24a.75.75 0 0 1 0 1.06Z" clipRule="evenodd" /></svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-3.5 h-3.5"><path fillRule="evenodd" d="M5.23 7.23a.75.75 0 0 1 1.06 0L10 10.94l3.71-3.71a.75.75 0 0 1 1.06 1.06l-4.24 4.24a.75.75 0 0 1-1.06 0L5.23 8.29a.75.75 0 0 1 0-1.06Z" clipRule="evenodd" /></svg>
            )}
          </button>
        </div>
      </Tooltip.Provider>

      {detailOpen && (
        <div className="mt-3 rounded-md border border-[#eceffd] bg-[#f7f8fd] px-3 py-2.5">
          <div className="space-y-1.5 text-xs text-[#2a3358]">
            <div className="flex justify-between">
              <span className="text-gray-500">场景权重</span>
              <span className="font-medium">{formatNumber(tp.platformWeight)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">业务线权重</span>
              <span className="font-medium">{formatNumber(tp.resourceGroupWeight)}</span>
            </div>

            <div className="border-t border-[#e0e4f0] my-1.5" />

            <div className="flex justify-between">
              <span className="text-gray-500">业务线优先级</span>
              <span className="font-medium">{formatNumber(tp.resourceGroupPriority)}</span>
            </div>

            <div className="ml-3 space-y-1.5">
              <div className="flex justify-between">
                <span className="text-gray-500">静态优先级</span>
                <span className="font-medium">{formatNumber(tp.bizWeight)} × {formatNumber(tp.bizPriority)} = {formatNumber(staticPriority)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">资源利用率系数</span>
                <span className="font-medium">{formatNumber(tp.gpuUtilWeight)} × {formatNumber(tp.gpuUtilPriority)} = {formatNumber(utilizationCoeff)}</span>
              </div>
            </div>

            <div className="border-t border-[#e0e4f0] my-1.5" />

            <div className="flex justify-between">
              <span className="text-gray-500">工作负载权重</span>
              <span className="font-medium">{formatNumber(tp.workloadWeight)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">工作负载优先级</span>
              <span className="font-medium">{formatNumber(tp.workloadPriority)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

registerProvisionDetailRenderer('tce', OctagramProvisionDetailRenderer);

export default OctagramProvisionDetailRenderer;
