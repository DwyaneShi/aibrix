import type {
  CreateJobResourceFieldsProps,
  CreateJobResourcePlugin,
} from './registry';
import { registerCreateJobResourcePlugin } from './registry';

export const DEFAULT_RESOURCE_PLUGIN = 'default';

export function ServiceReplicasField({
  replicas,
  replicaLimits,
  replicaError,
  onReplicasChange,
}: Pick<
  CreateJobResourceFieldsProps,
  'replicas' | 'replicaLimits' | 'replicaError' | 'onReplicasChange'
>) {
  return (
    <div>
      <label className="block text-sm mb-1">Service replicas</label>
      <p className="min-h-10 text-xs text-gray-400 mb-1">
        {replicaLimits
          ? `${replicaLimits.minReplicas} - ${replicaLimits.maxReplicas}`
          : 'Loading limits...'}
      </p>
      <input
        type="text"
        value={replicas}
        onChange={(event) => onReplicasChange(event.target.value)}
        placeholder="1"
        className={`w-full px-4 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-teal-500/30 focus:border-teal-500 bg-white ${
          replicaError ? 'border-red-300' : 'border-gray-200'
        }`}
      />
      {replicaError && (
        <p className="text-xs text-red-500 mt-1">{replicaError}</p>
      )}
    </div>
  );
}

export function DefaultRenderer(props: CreateJobResourceFieldsProps) {
  return <ServiceReplicasField {...props} />;
}

export function DefaultSettingsGuide() {
  return (
    <>
      <h4 className="mt-4 mb-2">Resource Configuration</h4>
      <div>
        <div className="mb-1">Service replicas:</div>
        <p className="text-gray-500">The number of dedicated workers to provision for this batch.</p>
      </div>
    </>
  );
}

export function createDefaultResourcePlugin(
  Fields: CreateJobResourcePlugin['Fields'],
): CreateJobResourcePlugin {
  return {
    Fields,
    SettingsGuide: DefaultSettingsGuide,
  };
}

const DefaultResourceFields = DefaultRenderer;
registerCreateJobResourcePlugin(
  DEFAULT_RESOURCE_PLUGIN,
  createDefaultResourcePlugin(DefaultResourceFields),
);
