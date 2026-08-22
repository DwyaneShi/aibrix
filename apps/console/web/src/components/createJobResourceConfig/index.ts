export {
  getCreateJobResourcePlugin,
  registerCreateJobResourcePlugin,
  type CreateJobResourceConfig,
  type CreateJobResourceFieldsProps,
  type CreateJobResourcePlugin,
} from './registry';

// Built-in plugins self-register through side-effect imports. Import the
// default first so every unknown backend has a resource configuration.
import './DefaultRenderer';
import './DemoRenderer';

// Downstream plugins are registered through side-effect imports here.
import './extension';
