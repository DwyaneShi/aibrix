import { useState, useMemo } from 'react';
import { ChevronDown, ChevronRight, ExternalLink } from 'lucide-react';
import { registerTimelineEventDetailRenderer, TimelineEventDetailProps } from './registry';
import type { MatchingOrderTimelineEntry } from './octagramTypes';

interface ParsedTimelineDetail {
  matchId?: string;
  matchOrderUrl?: string;
  timeline?: MatchingOrderTimelineEntry[];
}

function parseOctagramDetail(data: unknown): ParsedTimelineDetail | null {
  if (!data || typeof data !== 'object') return null;
  const d = data as Record<string, unknown>;
  if (!d.matchId && !d.timeline) return null;
  return {
    matchId: typeof d.matchId === 'string' ? d.matchId : undefined,
    matchOrderUrl: typeof d.matchOrderUrl === 'string' ? d.matchOrderUrl : undefined,
    timeline: Array.isArray(d.timeline) ? d.timeline : [],
  };
}

const DOT_PALETTE = [
  'bg-blue-500',
  'bg-violet-500',
  'bg-cyan-500',
  'bg-orange-500',
  'bg-teal-500',
  'bg-pink-500',
  'bg-indigo-500',
  'bg-amber-500',
  'bg-sky-500',
  'bg-fuchsia-500',
  'bg-lime-600',
  'bg-yellow-500',
] as const;

const BADGE_PALETTE = [
  'bg-blue-50 text-blue-700',
  'bg-violet-50 text-violet-700',
  'bg-cyan-50 text-cyan-700',
  'bg-orange-50 text-orange-700',
  'bg-teal-50 text-teal-700',
  'bg-pink-50 text-pink-700',
  'bg-indigo-50 text-indigo-700',
  'bg-amber-50 text-amber-700',
  'bg-sky-50 text-sky-700',
  'bg-fuchsia-50 text-fuchsia-700',
  'bg-lime-50 text-lime-700',
  'bg-yellow-50 text-yellow-700',
] as const;

const DOT_CANCELLED = 'bg-gray-400';
const DOT_CANCELLING = 'bg-gray-300';
const DOT_FAILED = 'bg-red-500';
const DOT_SUCCEEDED = 'bg-emerald-500';

const BADGE_CANCELLED = 'bg-gray-100 text-gray-500';
const BADGE_CANCELLING = 'bg-gray-50 text-gray-600';
const BADGE_FAILED = 'bg-red-50 text-red-700';
const BADGE_SUCCEEDED = 'bg-emerald-50 text-emerald-700';

function hasWord(status: string, word: string): boolean {
  return status.toLowerCase().includes(word);
}

function dotColor(status: string, index: number): string {
  if (hasWord(status, 'cancelled') || hasWord(status, 'canceled')) return DOT_CANCELLED;
  if (hasWord(status, 'cancelling') || hasWord(status, 'canceling')) return DOT_CANCELLING;
  if (hasWord(status, 'fail')) return DOT_FAILED;
  if (hasWord(status, 'success') || hasWord(status, 'succeeded')) return DOT_SUCCEEDED;
  if (hasWord(status, 'in_effect')) return DOT_SUCCEEDED;
  return DOT_PALETTE[index % DOT_PALETTE.length] ?? DOT_PALETTE[0];
}

function badgeCls(status: string, index: number): string {
  if (hasWord(status, 'cancelled') || hasWord(status, 'canceled')) return BADGE_CANCELLED;
  if (hasWord(status, 'cancelling') || hasWord(status, 'canceling')) return BADGE_CANCELLING;
  if (hasWord(status, 'fail')) return BADGE_FAILED;
  if (hasWord(status, 'success') || hasWord(status, 'succeeded')) return BADGE_SUCCEEDED;
  if (hasWord(status, 'in_effect')) return BADGE_SUCCEEDED;
  return BADGE_PALETTE[index % BADGE_PALETTE.length] ?? BADGE_PALETTE[0];
}

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function TimelineEntry({
  entry,
  isLast,
  dotColor,
  badgeCls,
}: {
  entry: MatchingOrderTimelineEntry;
  isLast: boolean;
  dotColor: string;
  badgeCls: string;
}) {
  return (
    <div className="flex gap-3 relative">
      <div className="flex flex-col items-center">
        <div className={`w-2.5 h-2.5 rounded-full shrink-0 mt-1 ${dotColor}`} />
        {!isLast && <div className="w-px flex-1 bg-gray-200 mt-1" />}
      </div>
      <div className={isLast ? 'pb-1' : 'pb-4'}>
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-sm text-gray-800">{entry.event}</span>
          <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${badgeCls}`}>
            {entry.new_display_status}
          </span>
        </div>
        {entry.note && <p className="text-xs text-gray-500 mt-0.5">{entry.note}</p>}
        <p className="text-xs text-gray-400 mt-0.5">{formatTime(entry.created_at)}</p>
      </div>
    </div>
  );
}

function OctagramTimelineRenderer({ data, jobStatus }: TimelineEventDetailProps) {
  const parsed = parseOctagramDetail(data);
  if (!parsed) return null;

  const autoExpand = jobStatus === 'resource_preparing' || jobStatus === 'resource_failed';
  const [expanded, setExpanded] = useState(autoExpand);

  const rawTimeline = parsed.timeline || [];

  const timeline = useMemo(() => {
    if (rawTimeline.length <= 1) return rawTimeline;
    const sorted = [...rawTimeline].sort(
      (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
    );
    return sorted;
  }, [rawTimeline]);

  return (
    <div className="ml-6 pt-1 pb-0 mt-2">
      <div className="flex items-center gap-2 mb-1">
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="inline-flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700"
        >
          {expanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
          Octagram Matching Order
        </button>

        {parsed.matchOrderUrl && (
          <a
            href={parsed.matchOrderUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-blue-500 hover:text-blue-600"
          >
            <ExternalLink className="w-3 h-3" />
            {parsed.matchId && (
              <span className="text-xs font-mono">#{parsed.matchId}</span>
            )}
          </a>
        )}

      </div>

      {expanded && timeline.length > 0 && (
        <div className="mt-1">
          {timeline.map((entry, i) => (
            <TimelineEntry
              key={i}
              entry={entry}
              isLast={i === timeline.length - 1}
              dotColor={dotColor(entry.new_display_status, i)}
              badgeCls={badgeCls(entry.new_display_status, i)}
            />
          ))}
        </div>
      )}

      {expanded && timeline.length === 0 && (
        <p className="text-xs text-gray-400">No matching order events yet.</p>
      )}
    </div>
  );
}

registerTimelineEventDetailRenderer('tce', 'resource_preparing', OctagramTimelineRenderer);
registerTimelineEventDetailRenderer('tce', 'resource_failed', OctagramTimelineRenderer);

export default OctagramTimelineRenderer;
