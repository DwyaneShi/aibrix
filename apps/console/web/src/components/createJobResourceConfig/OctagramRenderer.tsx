import { useEffect, useState } from 'react';
import { CalendarClock, Clock3, Sparkles } from 'lucide-react';
import { listCatalogRegionsForAccelerators } from '../../utils/api';
import type {
  CreateJobResourceConfig,
  CreateJobResourceFieldsProps,
  CreateJobResourcePlugin,
} from './registry';
import { registerCreateJobResourcePlugin } from './registry';
import { ServiceReplicasField } from './DefaultRenderer';

export const DEFAULT_RESOURCE_DURATION = '1h';
export const DEFAULT_RESOURCE_TYPE = 'scheduled';

type OctagramResourceType = 'reserved' | 'scheduled' | 'spot';

const RESOURCE_TYPES: Array<{
  value: OctagramResourceType;
  label: string;
  description: string;
  icon: typeof CalendarClock;
  supported: boolean;
}> = [
  {
    value: 'reserved',
    label: 'Reserved',
    description: 'Use reserved capacity.',
    icon: CalendarClock,
    supported: false,
  },
  {
    value: 'scheduled',
    label: 'Scheduled',
    description: 'Reserve capacity for a specific time window.',
    icon: Clock3,
    supported: true,
  },
  {
    value: 'spot',
    label: 'Spot',
    description: 'Use preemptible capacity when available.',
    icon: Sparkles,
    supported: false,
  },
];

function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : [];
}

function resourceType(value: CreateJobResourceConfig): OctagramResourceType {
  const configured = asString(value.resourceType);
  return RESOURCE_TYPES.some((option) => option.value === configured)
    ? configured as OctagramResourceType
    : DEFAULT_RESOURCE_TYPE;
}

export function mergeRegionOptions(
  selected: string[],
  available: string[],
): string[] {
  return Array.from(new Set([...selected, ...available])).sort();
}

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

export function validateOctagramResourceConfig(
  value: CreateJobResourceConfig,
): string | null {
  const selectedType = resourceType(value);
  if (selectedType !== 'scheduled') {
    return `${RESOURCE_TYPES.find((option) => option.value === selectedType)?.label ?? selectedType} resources are not supported yet.`;
  }
  const psm = asString(value.psm).trim();
  if (psm === '') {
    return 'PSM is required for Scheduled resources.';
  }
  return null;
}

export function OctagramRenderer({
  completionWindow,
  acceleratorType,
  replicas,
  replicaLimits,
  replicaError,
  onReplicasChange,
  value,
  onChange,
}: CreateJobResourceFieldsProps) {
  const selectedType = resourceType(value);
  const duration = durationForCompletionWindow(value.duration, completionWindow);
  const psm = asString(value.psm);
  const selectedRegions = asStringArray(value.regions);
  const [catalogRegions, setCatalogRegions] = useState<string[]>([]);
  const [regionsLoading, setRegionsLoading] = useState(false);
  const [regionsError, setRegionsError] = useState<string | null>(null);
  const updateConfig = (patch: CreateJobResourceConfig) => {
    onChange((current) => ({ ...current, ...patch }));
  };

  useEffect(() => {
    if (selectedType !== 'scheduled' || !acceleratorType || acceleratorType.toUpperCase() === 'CPU') {
      setCatalogRegions([]);
      setRegionsError(null);
      setRegionsLoading(false);
      return;
    }

    let active = true;
    const shouldDefaultRegions = value.regionAccelerator !== acceleratorType;
    if (shouldDefaultRegions) {
      onChange((current) => ({
        ...current,
        regionAccelerator: acceleratorType,
        regions: [],
      }));
    }
    setRegionsLoading(true);
    setRegionsError(null);
    void listCatalogRegionsForAccelerators([acceleratorType])
      .then((regionsByAccelerator) => {
        if (!active) return;
        const regions = regionsByAccelerator[acceleratorType] ?? [];
        setCatalogRegions(regions);
        if (shouldDefaultRegions) {
          onChange((current) => ({ ...current, regions: [...regions] }));
        }
      })
      .catch(() => {
        if (active) {
          setCatalogRegions([]);
          setRegionsError('Failed to load available regions.');
        }
      })
      .finally(() => {
        if (active) setRegionsLoading(false);
      });
    return () => {
      active = false;
    };
  // regionAccelerator deliberately stays out of providerConfig and only
  // tracks which accelerator the local region selection belongs to.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [acceleratorType, selectedType]);

  const regionOptions = mergeRegionOptions(selectedRegions, catalogRegions);
  const setType = (nextType: OctagramResourceType) => {
    updateConfig({ resourceType: nextType });
  };
  const toggleRegion = (region: string) => {
    updateConfig({
      regions: selectedRegions.includes(region)
        ? selectedRegions.filter((item) => item !== region)
        : [...selectedRegions, region],
    });
  };

  return (
    <div className="md:col-span-2 space-y-4">
      <div>
        <label className="block text-sm mb-2">Resource type</label>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {RESOURCE_TYPES.map((option) => {
            const Icon = option.icon;
            const selected = selectedType === option.value;
            return (
              <button
                key={option.value}
                type="button"
                onClick={() => setType(option.value)}
                className={`relative rounded-xl border px-4 py-3 text-left transition-colors ${
                  selected
                    ? 'border-teal-500 bg-teal-50/60 ring-1 ring-teal-500'
                    : 'border-gray-200 bg-white hover:border-gray-300'
                }`}
              >
                <div className="flex min-h-6 items-center gap-2 pr-14">
                  <Icon className={`w-4 h-4 ${selected ? 'text-teal-600' : 'text-gray-500'}`} />
                  <span className="text-sm font-medium">{option.label}</span>
                </div>
                {!option.supported && (
                  <span className="absolute right-3 top-1/2 -translate-y-1/2 rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[10px] font-medium text-amber-700">
                    Soon
                  </span>
                )}
              </button>
            );
          })}
        </div>
        <div className="mt-2 grid grid-cols-1 gap-3 sm:grid-cols-3">
          {RESOURCE_TYPES.map((option) => (
            <p
              key={`${option.value}-hint`}
              className={`px-1 text-xs leading-5 ${
                option.supported ? 'text-gray-500' : 'text-amber-700'
              }`}
            >
              {option.supported ? option.description : 'Stay tuned'}
            </p>
          ))}
        </div>
      </div>

      {selectedType === 'scheduled' && (
        <div className="space-y-4 rounded-xl border border-gray-100 bg-gray-50/50 p-4">
          <div>
            <label className="block text-sm mb-1">PSM <span className="text-red-500">*</span></label>
            <input
              type="text"
              value={psm}
              onChange={(event) => updateConfig({ psm: event.target.value })}
              placeholder="inf.aibrix.platform"
              className={`w-full px-4 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-teal-500/30 focus:border-teal-500 bg-white ${
                psm.trim() === '' ? 'border-red-200' : 'border-gray-200'
              }`}
            />
            {psm.trim() === '' && (
              <p className="text-xs text-red-500 mt-1">PSM is required.</p>
            )}
          </div>

          <div>
            <label className="block text-sm mb-1">Region</label>
            <p className="text-xs text-gray-400 mb-2">Select one or more datacenters that can run this job.</p>
            {regionsLoading ? (
              <p className="text-xs text-gray-400">Loading available regions...</p>
            ) : regionsError ? (
              <p className="text-xs text-amber-600">{regionsError}</p>
            ) : regionOptions.length === 0 ? (
              <p className="text-xs text-gray-400">No region constraints are available for this accelerator.</p>
            ) : (
              <div className="flex flex-wrap items-center gap-2">
                <label className="inline-flex items-center gap-1.5 px-2 py-1.5 text-xs text-gray-700">
                  <input
                    type="checkbox"
                    checked={selectedRegions.length === regionOptions.length}
                    onChange={(event) => updateConfig({
                      regions: event.target.checked ? [...regionOptions] : [],
                    })}
                    className="h-4 w-4 rounded border-gray-300 text-teal-600 focus:ring-teal-500"
                  />
                  All
                </label>
                {regionOptions.map((region) => {
                  const active = selectedRegions.includes(region);
                  return (
                    <button
                      key={region}
                      type="button"
                      onClick={() => toggleRegion(region)}
                      className={`px-3 py-1.5 text-xs rounded-full border transition-colors ${
                        active
                          ? 'bg-teal-600 text-white border-teal-600'
                          : 'bg-white text-gray-600 border-gray-200 hover:bg-gray-50'
                      }`}
                    >
                      {region}
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <ServiceReplicasField
              replicas={replicas}
              replicaLimits={replicaLimits}
              replicaError={replicaError}
              onReplicasChange={onReplicasChange}
            />
            <div>
              <label className="block text-sm mb-1">Duration</label>
              <p className="min-h-10 text-xs text-gray-400 mb-1">Continuous resource time required within the completion window.</p>
              <select
                value={duration}
                onChange={(event) => updateConfig({ duration: event.target.value })}
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
        </div>
      )}
    </div>
  );
}

export function OctagramSettingsGuide() {
  return (
    <>
      <h4 className="mt-4 mb-2">Resource Configuration</h4>
      <div>
        <div className="mb-1">Resource type:</div>
        <p className="text-gray-500">Scheduled reserves capacity for a specific time window. Reserved and Spot are coming soon.</p>
      </div>
      <div>
        <div className="mb-1">PSM:</div>
        <p className="text-gray-500">Identifies the PSM used for resource provision.</p>
      </div>
      <div>
        <div className="mb-1">Region:</div>
        <p className="text-gray-500">Limits scheduling to the selected datacenters that offer the template accelerator.</p>
      </div>
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

export function createOctagramResourcePlugin(
  Fields: CreateJobResourcePlugin['Fields'],
): CreateJobResourcePlugin {
  return {
    Fields,
    SettingsGuide: OctagramSettingsGuide,
    normalize: (value, completionWindow) => ({
      ...value,
      resourceType: resourceType(value),
      duration: durationForCompletionWindow(value.duration, completionWindow),
    }),
    validate: validateOctagramResourceConfig,
    toProviderConfig: (value, completionWindow) => ({
      resourceType: resourceType(value),
      psm: asString(value.psm).trim(),
      regions: asStringArray(value.regions),
      duration: durationForCompletionWindow(
        value.duration ?? DEFAULT_RESOURCE_DURATION,
        completionWindow,
      ),
    }),
  };
}

const OctagramResourceFields = OctagramRenderer;
registerCreateJobResourcePlugin(
  'octagram',
  createOctagramResourcePlugin(OctagramResourceFields),
);
registerCreateJobResourcePlugin(
  'tce',
  createOctagramResourcePlugin(OctagramResourceFields),
);
