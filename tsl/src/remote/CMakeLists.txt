set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/async.c
    ${CMAKE_CURRENT_SOURCE_DIR}/connection.c
    ${CMAKE_CURRENT_SOURCE_DIR}/connection_cache.c
    ${CMAKE_CURRENT_SOURCE_DIR}/copy_fetcher.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cursor_fetcher.c
    ${CMAKE_CURRENT_SOURCE_DIR}/data_fetcher.c
    ${CMAKE_CURRENT_SOURCE_DIR}/data_format.c
    ${CMAKE_CURRENT_SOURCE_DIR}/dist_txn.c
    ${CMAKE_CURRENT_SOURCE_DIR}/dist_commands.c
    ${CMAKE_CURRENT_SOURCE_DIR}/dist_copy.c
    ${CMAKE_CURRENT_SOURCE_DIR}/dist_ddl.c
    ${CMAKE_CURRENT_SOURCE_DIR}/prepared_statement_fetcher.c
    ${CMAKE_CURRENT_SOURCE_DIR}/copy_fetcher.c
    ${CMAKE_CURRENT_SOURCE_DIR}/healthcheck.c
    ${CMAKE_CURRENT_SOURCE_DIR}/stmt_params.c
    ${CMAKE_CURRENT_SOURCE_DIR}/tuplefactory.c
    ${CMAKE_CURRENT_SOURCE_DIR}/txn.c
    ${CMAKE_CURRENT_SOURCE_DIR}/txn_id.c
    ${CMAKE_CURRENT_SOURCE_DIR}/txn_resolve.c
    ${CMAKE_CURRENT_SOURCE_DIR}/txn_store.c
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.c)
target_sources(${TSL_LIBRARY_NAME} PRIVATE ${SOURCES})
target_include_directories(${TSL_LIBRARY_NAME}
                           PRIVATE ${PG_INCLUDEDIR} ${PG_INCLUDEDIR}/postgresql)
