import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "./client";
import type {
  ActionsResponse,
  MetricsSummary,
  Options,
  RunRecord,
  RunsResponse,
  ScalarPoint,
  VizEpisode,
  VizLayout,
  VizMeta,
} from "./types";

const ACTIVE = (status?: string) => status === "running" || status === "queued";

export function useOptions() {
  return useQuery<Options>({
    queryKey: ["options"],
    queryFn: () => api.get("/api/options"),
    staleTime: 30_000,
  });
}

export function useEnvConfigDefaults(name: string | undefined) {
  return useQuery<{ name: string; params: Record<string, unknown> }>({
    queryKey: ["env-config", name],
    queryFn: () => api.get(`/api/options/env-config/${name}`),
    enabled: Boolean(name),
    staleTime: 30_000,
  });
}

export function useRuns() {
  return useQuery<RunsResponse>({
    queryKey: ["runs"],
    queryFn: () => api.get("/api/runs"),
    refetchInterval: (query) =>
      query.state.data?.runs.some((run) => ACTIVE(run.status)) ? 2000 : 10_000,
  });
}

export function useRun(runId: string | undefined) {
  return useQuery<RunRecord>({
    queryKey: ["run", runId],
    queryFn: () => api.get(`/api/runs/${runId}`),
    enabled: Boolean(runId),
    refetchInterval: (query) => (ACTIVE(query.state.data?.status) ? 1500 : false),
  });
}

export function useMetricsSummary(runId: string | undefined, active: boolean) {
  return useQuery<MetricsSummary>({
    queryKey: ["metrics", runId],
    queryFn: () => api.get(`/api/runs/${runId}/metrics`),
    enabled: Boolean(runId),
    refetchInterval: active ? 3000 : false,
    retry: false,
  });
}

export function useScalars(
  runId: string | undefined,
  tags: string[],
  active: boolean,
) {
  const tagKey = tags.join(",");
  return useQuery<{ series: Record<string, ScalarPoint[]> }>({
    queryKey: ["scalars", runId, tagKey],
    queryFn: () =>
      api.get(`/api/runs/${runId}/metrics/scalars?tags=${encodeURIComponent(tagKey)}`),
    enabled: Boolean(runId) && tags.length > 0,
    refetchInterval: active ? 2000 : false,
    retry: false,
  });
}

export function useLogs(runId: string | undefined, active: boolean) {
  return useQuery<{ offset_next: number; content: string }>({
    queryKey: ["logs", runId],
    queryFn: () => api.get(`/api/runs/${runId}/logs`),
    enabled: Boolean(runId),
    refetchInterval: active ? 2000 : false,
  });
}

export function useCheckpoints(runId: string | undefined) {
  return useQuery<{ agents: string[]; checkpoints: (string | number)[] }>({
    queryKey: ["checkpoints", runId],
    queryFn: () => api.get(`/api/runs/${runId}/checkpoints`),
    enabled: Boolean(runId),
  });
}

export function useActions(runId: string | undefined, episode: number | undefined) {
  return useQuery<ActionsResponse>({
    queryKey: ["actions", runId, episode],
    queryFn: () =>
      api.get(
        `/api/runs/${runId}/actions${episode !== undefined ? `?episode=${episode}` : ""}`,
      ),
    enabled: Boolean(runId),
    retry: false,
  });
}

export function useVizMeta(runId: string | undefined, active: boolean) {
  return useQuery<VizMeta>({
    queryKey: ["viz-meta", runId],
    queryFn: () => api.get(`/api/runs/${runId}/viz/meta`),
    enabled: Boolean(runId),
    refetchInterval: active ? 3000 : false,
    retry: false,
  });
}

export function useVizLayout(runId: string | undefined, enabled: boolean) {
  return useQuery<VizLayout>({
    queryKey: ["viz-layout", runId],
    queryFn: () => api.get(`/api/runs/${runId}/viz/layout`),
    enabled,
    staleTime: Infinity,
    retry: false,
  });
}

export function useVizEpisode(
  runId: string | undefined,
  episode: number | undefined,
  enabled: boolean,
) {
  return useQuery<VizEpisode>({
    queryKey: ["viz-episode", runId, episode],
    queryFn: () => api.get(`/api/runs/${runId}/viz/episodes/${episode}`),
    enabled: enabled && episode !== undefined,
    staleTime: Infinity,
    retry: false,
  });
}

export function useStopRun() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (runId: string) => api.post<RunRecord>(`/api/runs/${runId}/stop`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["runs"] });
      queryClient.invalidateQueries({ queryKey: ["run"] });
    },
  });
}

export function useDeleteRun() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ runId, artifacts }: { runId: string; artifacts: boolean }) =>
      api.delete(`/api/runs/${runId}?artifacts=${artifacts}`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["runs"] }),
  });
}

export function useLaunchTrain() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: {
      display_name: string;
      base_config: string;
      params: Record<string, unknown>;
    }) => api.post<RunRecord>("/api/runs/train", body),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["runs"] }),
  });
}

export function useLaunchEvaluate() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: {
      display_name: string;
      base_config: string;
      source: { run_id?: string; experiment_name?: string };
      checkpoint: string;
      params: Record<string, unknown>;
    }) => api.post<RunRecord>("/api/runs/evaluate", body),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["runs"] }),
  });
}
