export interface TicketPriorityDetail {
  ticketId: number;
  priority: number;
  resourceGroupPriority: number;
  resourceGroupWeight: number;
  gpuUtilPriority: number;
  gpuUtilWeight: number;
  bizPriority: number;
  bizWeight: number;
  workloadPriority: number;
  workloadWeight: number;
  sceneWeight: number;
  platformWeight: number;
  workloadPriorityDescCn?: string;
  workloadPriorityDescI18n?: string;
}
