import type { ComponentType, Dispatch, SetStateAction } from 'react';
import type { CompletionWindowOption } from '../../utils/batchProduct';

export type CreateJobResourceConfig = Record<string, unknown>;

export interface CreateJobResourceFieldsProps {
  completionWindow: CompletionWindowOption;
  acceleratorType?: string;
  replicas: string;
  replicaLimits?: {
    minReplicas: number;
    maxReplicas: number;
  };
  replicaError?: string;
  onReplicasChange: (value: string) => void;
  value: CreateJobResourceConfig;
  onChange: Dispatch<SetStateAction<CreateJobResourceConfig>>;
}

export interface CreateJobResourcePlugin {
  Fields: ComponentType<CreateJobResourceFieldsProps>;
  SettingsGuide?: ComponentType;
  normalize?: (
    value: CreateJobResourceConfig,
    completionWindow: CompletionWindowOption,
  ) => CreateJobResourceConfig;
  toProviderConfig?: (
    value: CreateJobResourceConfig,
    completionWindow: CompletionWindowOption,
  ) => CreateJobResourceConfig;
  validate?: (
    value: CreateJobResourceConfig,
    completionWindow: CompletionWindowOption,
  ) => string | null;
}

const registry = new Map<string, CreateJobResourcePlugin>();
const defaultPluginKey = 'default';

export function registerCreateJobResourcePlugin(
  provider: string,
  plugin: CreateJobResourcePlugin,
) {
  registry.set(provider.toLowerCase(), plugin);
}

export function getCreateJobResourcePlugin(
  provider: string,
): CreateJobResourcePlugin {
  const plugin = registry.get(provider.toLowerCase()) ?? registry.get(defaultPluginKey);
  if (!plugin) {
    throw new Error('default create-job resource plugin is not registered');
  }
  return plugin;
}
