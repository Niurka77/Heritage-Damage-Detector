-- ============================================================
--  HERITAGE DAMAGE DETECTOR v6.5
--  SCRIPT DE CONFIGURACION PARA HISTORIAL CON IMAGENES
--  (Bucket de Storage + Politicas RLS + Columnas nuevas)
--  Proyecto: gosywjlocfoettpzafga
--  Instrucciones: pega TODO este bloque en el SQL Editor de
--  Supabase (Dashboard -> SQL Editor) y pulsa "Run".
-- ============================================================

-- 1) CREAR EL BUCKET PUBLICO "inspecciones" (imagenes del historial)
insert into storage.buckets (id, name, public)
values ('inspecciones', 'inspecciones', true)
on conflict (id) do nothing;

-- 2) POLITICAS DE SEGURIDAD (RLS) para que la app (rol anon) pueda
--    guardar y leer las imagenes dentro del bucket "inspecciones".

-- 2a) Permitir a cualquiera (anon) SUBIR imagenes al bucket
drop policy if exists "anon_insert_inspecciones" on storage.objects;
create policy "anon_insert_inspecciones"
on storage.objects for insert
to anon
with check (bucket_id = 'inspecciones');

-- 2b) Permitir a cualquiera (anon) LEER/DESCARGAR imagenes del bucket
drop policy if exists "anon_select_inspecciones" on storage.objects;
create policy "anon_select_inspecciones"
on storage.objects for select
to anon
using (bucket_id = 'inspecciones');

-- 2c) Permitir a cualquiera (anon) ACTUALIZAR (por si se re-guarda)
drop policy if exists "anon_update_inspecciones" on storage.objects;
create policy "anon_update_inspecciones"
on storage.objects for update
to anon
using (bucket_id = 'inspecciones')
with check (bucket_id = 'inspecciones');

-- 2d) Permitir a cualquiera (anon) ELIMINAR (por si hay que limpiar)
drop policy if exists "anon_delete_inspecciones" on storage.objects;
create policy "anon_delete_inspecciones"
on storage.objects for delete
to anon
using (bucket_id = 'inspecciones');

-- 3) AÑADIR COLUMNAS PARA GUARDAR LAS URLS DE LAS IMAGENES
--    en la tabla principal de inspecciones (si no existen ya).
alter table public.inspection_results
  add column if not exists imagen_original_url text;
alter table public.inspection_results
  add column if not exists imagen_anotada_url text;

-- Verificacion rapida (opcional): muestra el bucket creado
select id, name, public from storage.buckets where id = 'inspecciones';

-- ============================================================
--  FIN DEL SCRIPT - Si todo salio bien veras una fila con
--  id='inspecciones', name='inspecciones', public=true
-- ============================================================
