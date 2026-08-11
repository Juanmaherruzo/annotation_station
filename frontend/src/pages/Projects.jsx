import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getProjects, createProject, deleteProject } from "../api";

export default function Projects() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [name, setName] = useState("");
  const [taskType, setTaskType] = useState("instance_segmentation");

  const { data: projects = [], isLoading } = useQuery({
    queryKey: ["projects"],
    queryFn: getProjects,
  });

  const create = useMutation({
    mutationFn: () => createProject({ name, task_type: taskType }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["projects"] });
      setName("");
    },
  });

  const remove = useMutation({
    mutationFn: (id) => deleteProject(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["projects"] }),
  });

  return (
    <div className="min-h-screen bg-surface p-8">
      <h1 className="text-3xl font-bold mb-8">SAMark</h1>

      {/* New project form */}
      <div className="bg-panel rounded-lg p-6 mb-8 max-w-md">
        <h2 className="text-lg font-semibold mb-4">New project</h2>
        <input
          className="w-full bg-accent rounded px-3 py-2 mb-3 text-white"
          placeholder="Project name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <select
          className="w-full bg-accent rounded px-3 py-2 mb-4 text-white"
          value={taskType}
          onChange={(e) => setTaskType(e.target.value)}
        >
          <option value="instance_segmentation">Instance segmentation</option>
          <option value="object_detection">Object detection</option>
        </select>
        <button
          className="w-full bg-blue-600 hover:bg-blue-500 rounded py-2 font-medium disabled:opacity-50"
          disabled={!name.trim() || create.isPending}
          onClick={() => create.mutate()}
        >
          Create
        </button>
      </div>

      {/* Project list */}
      {isLoading ? (
        <p className="text-gray-400">Loading...</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 max-w-4xl">
          {projects.map((p) => (
            <div
              key={p.id}
              className="bg-panel rounded-lg p-5 cursor-pointer hover:ring-2 hover:ring-blue-500 transition"
              onClick={() => navigate(`/projects/${p.id}/annotate`)}
            >
              <div className="flex justify-between items-start">
                <div>
                  <p className="font-semibold text-lg">{p.name}</p>
                  <p className="text-xs text-gray-400 mt-1">{p.task_type}</p>
                  <p className="text-xs text-gray-400">
                    {p.annotated_count}/{p.image_count} images annotated
                  </p>
                </div>
                <button
                  className="text-gray-500 hover:text-red-400 text-sm ml-2"
                  onClick={(e) => {
                    e.stopPropagation();
                    remove.mutate(p.id);
                  }}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
