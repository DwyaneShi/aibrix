import { describe, expect, it } from 'vitest';
import { getCreateJobResourcePlugin } from '.';
import { DefaultRenderer, DefaultSettingsGuide } from './DefaultRenderer';
import { DemoRenderer, DemoSettingsGuide } from './DemoRenderer';

describe('create job resource plugin registry', () => {
  it('uses the default replicas plugin for an unregistered backend', () => {
    const plugin = getCreateJobResourcePlugin('unregistered-provider');

    expect(plugin.Fields).toBe(DefaultRenderer);
    expect(plugin.SettingsGuide).toBe(DefaultSettingsGuide);
  });

  it('prefers a registered backend plugin over the default', () => {
    const plugin = getCreateJobResourcePlugin('demo');

    expect(plugin.Fields).toBe(DemoRenderer);
    expect(plugin.SettingsGuide).toBe(DemoSettingsGuide);
  });
});
