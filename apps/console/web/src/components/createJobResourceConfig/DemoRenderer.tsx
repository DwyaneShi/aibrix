import type {
  CreateJobResourceFieldsProps,
  CreateJobResourcePlugin,
} from './registry';
import { registerCreateJobResourcePlugin } from './registry';
import { ServiceReplicasField } from './DefaultRenderer';

export const DEFAULT_RESOURCE_DURATION = '1h';

export function durationForCompletionWindow(
  duration: unknown,
  completionWindow: string,
): string {
  const completionHours = parseInt(completionWindow, 10);
  const durationHours = typeof duration === 'string'
    ? parseInt(duration, 10)
    : 1;
  const normalizedHours = Number.isFinite(durationHours)
    ? Math.min(Math.max(durationHours, 1), completionHours)
    : 1;
  return `${normalizedHours}h`;
}

export function DemoRenderer({
  completionWindow,
  replicas,
  replicaLimits,
  replicaError,
  onReplicasChange,
  value,
  onChange,
}: CreateJobResourceFieldsProps) {
  const duration = durationForCompletionWindow(value.duration, completionWindow);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <ServiceReplicasField
        replicas={replicas}
        replicaLimits={replicaLimits}
        replicaError={replicaError}
        onReplicasChange={onReplicasChange}
      />
      <div>
        <label className="block text-sm mb-1">Duration</label>
        <p className="min-h-10 text-xs text-gray-400 mb-1">
          Continuous resource time required within the completion window.
        </p>
        <select
          value={duration}
          onChange={(event) => onChange({ ...value, duration: event.target.value })}
          className="w-full px-4 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-teal-500/30 focus:border-teal-500 bg-white"
        >
          {Array.from(
            { length: parseInt(completionWindow, 10) },
            (_, index) => index + 1,
          ).map(hours => (
            <option key={hours} value={`${hours}h`}>
              {hours} hr
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

export function DemoSettingsGuide() {
  return (
    <>
      <h4 className="mt-4 mb-2">Resource Configuration</h4>
      <div>
        <div className="mb-1">Service replicas:</div>
        <p className="text-gray-500">The number of dedicated workers to provision for this batch.</p>
      </div>
      <div>
        <div className="mb-1">Duration:</div>
        <p className="text-gray-500">The continuous resource time required within the completion window.</p>
      </div>
    </>
  );
}

export function createDemoResourcePlugin(
  Fields: CreateJobResourcePlugin['Fields'],
): CreateJobResourcePlugin {
  return {
    Fields,
    SettingsGuide: DemoSettingsGuide,
    normalize: (value, completionWindow) => ({
      ...value,
      duration: durationForCompletionWindow(value.duration, completionWindow),
    }),
    toProviderConfig: (value, completionWindow) => ({
      duration: durationForCompletionWindow(
        value.duration ?? DEFAULT_RESOURCE_DURATION,
        completionWindow,
      ),
    }),
  };
}

const DemoResourceFields = DemoRenderer;
registerCreateJobResourcePlugin(
  'demo',
  createDemoResourcePlugin(DemoResourceFields),
);
